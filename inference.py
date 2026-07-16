#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "loguru",
#     "torch>=2.8.0",
#     "torchvision",
#     "transformers>=5.0.0",
#     "accelerate",
#     "mcap-owa-support>=0.6.5",
#     "owa-core>=0.6.5",
#     "owa-msgs>=0.6.5",
#     "owa-env-desktop>=0.6.5",
#     "owa-data @ git+https://github.com/open-world-agents/open-world-agents@8fee481a65c719b8565a674de62966f955e911cf#subdirectory=projects/owa-data",
# ]
#
# [tool.uv]
# exclude-newer = "2026-05-08"
# ///
"""
Generalist-IDM inference script: extracts actions from video and outputs MCAP.

Usage:
    uv run inference.py input_video.mp4 output.mcap
    uv run inference.py input_video.mp4 output.mcap --model open-world-agents/Generalist-IDM-1B
    uv run inference.py input_video.mp4 output.mcap --device cpu --max-duration 30
"""

import argparse
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, cast

import torch
from loguru import logger
from transformers import AutoModelForImageTextToText, AutoProcessor

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.core import MESSAGES
from owa.data.encoders import EventEncoderError, FactorizedEventEncoder, HierarchicalEventEncoder, create_encoder
from owa.data.tokenization import (
    EventTokenizationContext,
    TokenizedEvent,
    decode_event,
    get_image_config,
    prepare_model_for_events,
    tokenize_event,
)
from owa.data.processing.resampler import EventResamplerDict

MODEL_ID = "open-world-agents/Generalist-IDM-1B"


@dataclass
class InferenceConfig:
    model_path: str
    device: str = "cuda"
    max_context_length: int = 1024
    max_new_tokens: int = 20
    screen_resample_rate_hz: float = 20.0
    mouse_resample_rate_hz: float = 20.0
    keyboard_resample_rate_hz: float = 0.0
    trust_remote_code: bool = True
    action_topics: list[str] = field(default_factory=lambda: ["keyboard", "mouse/raw"])
    time_shift_seconds_for_action: float | None = None


def get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    return float(result.stdout.strip())


def preprocess_video(input_path: str, output_path: str, duration: float) -> str:
    """Cut video to duration, resize to 448x448, and set keyframes using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-t",
        str(duration),
        "-vsync",
        "1",
        "-filter:v",
        "fps=60,scale=448:448",
        "-c:v",
        "libx264",
        "-x264-params",
        "keyint=30:no-scenecut=1:bframes=0",
        "-an",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr}")
    return output_path


def create_mcap_from_video(video_path: str, mcap_path: str, fps: float = 20.0):
    """Create MCAP file with screen events from video."""
    ScreenCaptured = MESSAGES["desktop/ScreenCaptured"]
    duration = get_video_duration(video_path)
    # Use absolute path so video can be found regardless of MCAP location
    video_abs_path = str(Path(video_path).resolve())

    with OWAMcapWriter(mcap_path) as writer:
        interval_ns = int(1e9 / fps)
        num_frames = int(duration * fps)
        for i in range(num_frames):
            timestamp_ns = i * interval_ns
            screen_msg = ScreenCaptured(
                utc_ns=timestamp_ns,
                media_ref={"uri": video_abs_path, "pts_ns": timestamp_ns},
            )
            writer.write_message(screen_msg, topic="screen", timestamp=timestamp_ns)
    logger.info(f"Created MCAP with {num_frames} frames at {fps} FPS")


def resample_event_stream(
    raw_events: Iterable[McapMessage],
    *,
    screen_resample_rate_hz: float,
    mouse_resample_rate_hz: float,
    keyboard_resample_rate_hz: float,
) -> Iterator[McapMessage]:
    """Resample raw events and yield all resampled events."""
    resampler = EventResamplerDict(
        {
            "screen": screen_resample_rate_hz,
            "mouse/raw": mouse_resample_rate_hz,
            "keyboard": keyboard_resample_rate_hz,
        }
    )
    for mcap_msg in raw_events:
        resampler.add_event(mcap_msg)
        resampler.step(mcap_msg.timestamp)
        for resampled_msg in resampler.pop_events():
            yield resampled_msg


class _ContextManager:
    """Manages context window for event generation with automatic trimming."""

    def __init__(
        self,
        *,
        device: str,
        max_context_length: int,
        processor_image_processor,
        tokenization_ctx: EventTokenizationContext,
        callback: Optional[Callable[[McapMessage], None]] = None,
    ):
        self.device = device
        self.callback = callback
        self.max_context_length = max_context_length
        self.processor_image_processor = processor_image_processor
        self.tokenization_ctx = tokenization_ctx
        self.sequences = torch.tensor([], dtype=torch.long, device=device)
        self.pixel_values = torch.tensor([], dtype=torch.bfloat16, device=device)
        self.event_indices = torch.tensor([], dtype=torch.long, device=device)
        self.image_counts = torch.tensor([], dtype=torch.long, device=device)
        self.last_timestamp = None
        self.timestamp_bias = 0

    def __repr__(self):
        return f"Context(seq_len={len(self.sequences)}, images={len(self.pixel_values)}, events={len(self.event_indices)})"

    def append_event(self, event: McapMessage, *, dry_run: bool = False, is_timestamp_adjusted: bool = False) -> int:
        tokenized_event = tokenize_event(self.tokenization_ctx, event)
        encoder = self.tokenization_ctx.encoder
        if not isinstance(encoder, (HierarchicalEventEncoder, FactorizedEventEncoder)):
            raise NotImplementedError(f"Encoder type {type(encoder)} is not supported.")
        timestamp_range = encoder.config.timestamp_range

        if is_timestamp_adjusted:
            timestamp_bias = event.timestamp - (event.timestamp % timestamp_range)
            adjusted_timestamp = event.timestamp
        else:
            timestamp_bias = self.timestamp_bias
            if self.last_timestamp is not None and (
                self.last_timestamp % timestamp_range > event.timestamp % timestamp_range
            ):
                timestamp_bias += timestamp_range
            adjusted_timestamp = event.timestamp + int(timestamp_bias)

        if dry_run:
            return adjusted_timestamp
        event.timestamp = adjusted_timestamp
        self.last_timestamp = event.timestamp
        self.timestamp_bias = timestamp_bias
        if self.callback:
            self.callback(event)
        self._append_tensors(tokenized_event)
        self._trim_if_needed()
        return event.timestamp

    def _append_tensors(self, tokenized_event: TokenizedEvent):
        self.event_indices = torch.cat([self.event_indices, torch.tensor([len(self.sequences)], device=self.device)])
        new_tokens = torch.tensor(tokenized_event["token_ids"], dtype=torch.long, device=self.device)
        self.sequences = torch.cat([self.sequences, new_tokens])
        new_images = tokenized_event["images"]
        self.image_counts = torch.cat([self.image_counts, torch.tensor([len(new_images)], device=self.device)])
        if new_images:
            pil_images = []
            for img in new_images:
                try:
                    pil_images.append(img.to_pil_image())
                except Exception as e:
                    from PIL import Image

                    logger.warning(f"Failed to load image: {e}. Using black placeholder.")
                    pil_images.append(Image.new("RGB", (448, 448), color="black"))
            pixel_values = self.processor_image_processor(pil_images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device, dtype=torch.bfloat16)
            self.pixel_values = torch.cat([self.pixel_values, pixel_values])

    def _trim_if_needed(self):
        while len(self.sequences) > self.max_context_length:
            self._pop_first_event()

    def _pop_first_event(self):
        if len(self.event_indices) <= 1:
            return
        second_event_start = self.event_indices[1].item()
        first_event_images = self.image_counts[0].item()
        self.sequences = self.sequences[second_event_start:]
        self.pixel_values = self.pixel_values[first_event_images:]
        self.event_indices = self.event_indices[1:] - second_event_start
        self.image_counts = self.image_counts[1:]


class InferencePipeline:
    """Prediction pipeline that reads MCAP, runs the model, and writes labeled MCAP."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_path,
            device_map=config.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=config.trust_remote_code,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(config.model_path, trust_remote_code=config.trust_remote_code)
        self.tokenizer = self.processor.tokenizer
        image_config = get_image_config(config.model_path)
        # NOTE: all published G-IDM checkpoints use factorized encoder
        encoder = create_encoder("factorized", fake_image_placeholder=image_config.fake_placeholder)
        prepare_model_for_events(self.tokenizer, encoder, image_config, self.model)
        self.tokenization_ctx = EventTokenizationContext(
            encoder=encoder, tokenizer=self.tokenizer, image_config=image_config
        )
        self._eos_token_id = self.tokenizer.encode("<EVENT_END>")[0]

    def _generate_single_event(self, sequences: torch.Tensor, pixel_values: torch.Tensor) -> torch.LongTensor:
        attention_mask = torch.ones_like(sequences, dtype=torch.bool, device=sequences.device)
        eos_tokens = [self._eos_token_id, self.tokenizer.convert_tokens_to_ids("<SCREEN>")]
        outputs = self.model.generate(
            input_ids=sequences,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            pad_token_id=self._eos_token_id,
            eos_token_id=eos_tokens,
            max_new_tokens=self.config.max_new_tokens,
            use_cache=True,
        )[0, sequences.shape[1] :]
        return cast(torch.LongTensor, outputs)

    def run_generation(
        self,
        input_iterator: Iterable[McapMessage],
        callback: Callable[[McapMessage], None],
        *,
        apply_resampler: bool = True,
    ):
        def output_event(event: McapMessage):
            if event.topic in self.config.action_topics and self.config.time_shift_seconds_for_action is not None:
                event.timestamp = max(0, event.timestamp - int(self.config.time_shift_seconds_for_action * 1e9))
            callback(event)

        context = _ContextManager(
            device=self.config.device,
            max_context_length=self.config.max_context_length,
            processor_image_processor=self.processor.image_processor,
            tokenization_ctx=self.tokenization_ctx,
            callback=output_event,
        )
        if apply_resampler:
            input_iterator = resample_event_stream(
                input_iterator,
                screen_resample_rate_hz=self.config.screen_resample_rate_hz,
                mouse_resample_rate_hz=self.config.mouse_resample_rate_hz,
                keyboard_resample_rate_hz=self.config.keyboard_resample_rate_hz,
            )
        input_iter = iter(input_iterator)
        try:
            first_event = next(input_iter)
        except StopIteration:
            return
        context.append_event(first_event, is_timestamp_adjusted=True)

        for next_event in input_iter:
            while True:
                sequences = context.sequences.unsqueeze(0)
                new_tokens = self._generate_single_event(sequences, context.pixel_values)
                try:
                    generated_event = decode_event(self.tokenization_ctx, new_tokens.cpu().numpy())
                except EventEncoderError:
                    logger.debug("Generated invalid event, stopping generation for this gap")
                    break
                # NOTE(claude): turn off following for format 2 and turn on for format 3
                # if (
                #     generated_event.topic in self.config.action_topics
                #     and self.config.time_shift_seconds_for_action is not None
                # ):
                #     generated_event.timestamp += int(self.config.time_shift_seconds_for_action * 1e9)
                if next_event.timestamp < context.append_event(generated_event, dry_run=True):
                    logger.debug("Generated event is after next input event, stop generating")
                    break
                context.append_event(generated_event)
                logger.success(f"Generated event: {generated_event.topic} at {generated_event.timestamp}, {context}")
            context.append_event(next_event, is_timestamp_adjusted=True)
            logger.info(f"Added input event: {next_event.topic} at {next_event.timestamp}, {context}")

    def pseudo_label_action(self, src_mcap_path: str, dst_mcap_path: str):
        if self.config.time_shift_seconds_for_action == 0:
            import warnings

            warnings.warn("Pseudo-labeling requires a non-zero time shift. Setting to 0.1s.")
            self.config.time_shift_seconds_for_action = 0.1

        def resolve_screen_paths():
            with OWAMcapReader(src_mcap_path) as reader:
                for mcap_msg in reader.iter_messages(topics=["screen"]):
                    if mcap_msg.topic == "screen":
                        mcap_msg.decoded.resolve_relative_path(src_mcap_path)
                    yield mcap_msg

        with OWAMcapWriter(dst_mcap_path) as writer:
            self.run_generation(resolve_screen_paths(), writer.write_message, apply_resampler=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generalist-IDM inference: extract actions from video to MCAP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run inference.py input.mp4 output.mcap
    uv run inference.py gameplay.mkv actions.mcap --device cpu
    uv run inference.py video.mp4 out.mcap --max-duration 60 --max-context-length 2048
""",
    )
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument("output_mcap", type=str, help="Path to output MCAP file")
    parser.add_argument("--model", type=str, default=MODEL_ID, help=f"Model path or HF ID (default: {MODEL_ID})")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (default: cuda)")
    parser.add_argument(
        "--max-duration", type=float, default=None, help="Max video duration in seconds (default: no limit)"
    )
    parser.add_argument("--max-context-length", type=int, default=2048, help="Max context length (default: 2048)")
    parser.add_argument(
        "--time-shift", type=float, default=0.1, help="Time shift for actions in seconds (default: 0.1)"
    )
    args = parser.parse_args()

    input_video = Path(args.input_video)
    output_mcap = Path(args.output_mcap)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Preprocess video if max_duration is specified
        if args.max_duration is not None:
            logger.info(f"Preprocessing video (max {args.max_duration}s)...")
            processed_video = str(tmpdir / "processed.mkv")
            preprocess_video(str(input_video), processed_video, args.max_duration)
        else:
            processed_video = str(input_video)

        # Create input MCAP from video
        logger.info("Creating MCAP from video...")
        input_mcap = str(tmpdir / "input.mcap")
        create_mcap_from_video(processed_video, input_mcap)

        # Run inference
        logger.info("Running IDM inference...")
        config = InferenceConfig(
            model_path=args.model,
            device=args.device,
            max_context_length=args.max_context_length,
            time_shift_seconds_for_action=args.time_shift,
        )
        pipeline = InferencePipeline(config)
        pipeline.pseudo_label_action(input_mcap, str(output_mcap))

    logger.success(f"Output written to: {output_mcap}")


if __name__ == "__main__":
    main()
