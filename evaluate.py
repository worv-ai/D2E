#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "loguru==0.7.2",
#     "numpy>=1.24.0",
#     "mcap-owa-support @ git+https://github.com/lastdefiance20/open-world-agents.git#subdirectory=projects/mcap-owa-support",
#     "owa-core @ git+https://github.com/lastdefiance20/open-world-agents.git#subdirectory=projects/owa-core",
#     "owa-msgs @ git+https://github.com/lastdefiance20/open-world-agents.git#subdirectory=projects/owa-msgs",
# ]
# ///
"""
G-IDM evaluation script: compare ground truth MCAP with predicted MCAP.

Usage:
    uv run evaluate.py ground_truth.mcap predicted.mcap
    uv run evaluate.py ground_truth.mcap predicted.mcap --bin-ms 50
    uv run evaluate.py ground_truth.mcap predicted.mcap --output results.json

Metrics (from D2E paper Section F.2):
    - Mouse: Pearson correlation (X/Y), Scale ratio (X/Y)
    - Keyboard: Per-key accuracy
    - Mouse buttons: Per-button accuracy
    - All metrics computed over non-overlapping 50ms temporal bins
"""

import argparse
import copy
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, TypedDict

import numpy as np
from loguru import logger

from mcap_owa.highlevel import OWAMcapReader
from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.msgs.desktop.mouse import RawMouseEvent


class EventType(StrEnum):
    """Type of event."""

    SCREEN = "screen"
    KEYBOARD = "keyboard"
    MOUSE_OP = "mouse_op"
    MOUSE_NOP = "mouse_nop"


def _determine_event_type(event: McapMessage) -> EventType:
    """Determine the event type from a decoded event object."""
    event_type = event.topic
    if event_type == "mouse/raw":
        if event.decoded.button_flags != 0:
            return EventType.MOUSE_OP
        else:
            return EventType.MOUSE_NOP
    if event_type in ["keyboard", "screen"]:
        return EventType(event_type)
    raise ValueError(f"Unknown event type: {event_type}")


@dataclass
class BinningConfig:
    bin_size_ns: int = 50_000_000  # 50ms bins (20Hz)
    empty_bins_as_correct: bool = False


class Bin(TypedDict):
    start_ts: int
    end_ts: int
    events: List[McapMessage]


def create_bins(start_time: int, end_time: int, bin_size_ns: int) -> List[Bin]:
    """Create time bins for the given time range."""
    bins = []
    current_time = start_time
    while current_time < end_time:
        bin_end = min(current_time + bin_size_ns, end_time)
        bins.append({"start_ts": current_time, "end_ts": bin_end, "events": []})
        current_time = bin_end
    return bins


def bin_events(events: Iterator[McapMessage], bins: List[Bin]) -> List[Bin]:
    """Assign events to appropriate time bins."""
    last_bin_idx = 0
    for event in events:
        while last_bin_idx < len(bins) and bins[last_bin_idx]["end_ts"] <= event.timestamp:
            last_bin_idx += 1
        if last_bin_idx >= len(bins):
            break
        bins[last_bin_idx]["events"].append(event)
    return bins


class MouseMoveMetric:
    """Mouse movement metrics: Pearson correlation and Scale ratio (X/Y)."""

    name = "mouse_move"

    def __init__(self, cfg: BinningConfig):
        self.empty_bins_as_correct = cfg.empty_bins_as_correct
        self.src_x_values: List[float] = []
        self.dst_x_values: List[float] = []
        self.src_y_values: List[float] = []
        self.dst_y_values: List[float] = []
        self.src_sequences: List[np.ndarray] = []
        self.dst_sequences: List[np.ndarray] = []

    def update(self, src_bin: Bin, dst_bin: Bin):
        """Update the metric with a new bin."""
        src_moves = self._extract_mouse_moves(src_bin)
        dst_moves = self._extract_mouse_moves(dst_bin)

        src_total = np.array(src_moves).sum(axis=0) if src_moves else np.array([0, 0])
        dst_total = np.array(dst_moves).sum(axis=0) if dst_moves else np.array([0, 0])

        # Only add to Pearson calculation if at least one has movement
        # (matches original metrics.py behavior)
        if src_moves or dst_moves:
            self.src_x_values.append(float(src_total[0]))
            self.dst_x_values.append(float(dst_total[0]))
            self.src_y_values.append(float(src_total[1]))
            self.dst_y_values.append(float(dst_total[1]))

        # Scale ratio always includes all bins
        self.src_sequences.append(src_total)
        self.dst_sequences.append(dst_total)

    def _extract_mouse_moves(self, bin_data: Bin) -> List[Tuple[int, int]]:
        """Extract mouse movement deltas from a bin."""
        moves = []
        for event in bin_data["events"]:
            if _determine_event_type(event) == EventType.MOUSE_NOP:
                decoded = event.decoded
                moves.append((decoded.dx, decoded.dy))
        return moves

    @property
    def metric(self) -> Dict[str, Any]:
        """Return the current metric value."""
        result = {}

        # Pearson correlation for X
        if len(self.src_x_values) > 1:
            try:
                corr_x = np.corrcoef(self.src_x_values, self.dst_x_values)[0, 1]
                result["pearson_x"] = float(corr_x) if not np.isnan(corr_x) else None
            except (ValueError, FloatingPointError):
                result["pearson_x"] = None
        else:
            result["pearson_x"] = None

        # Pearson correlation for Y
        if len(self.src_y_values) > 1:
            try:
                corr_y = np.corrcoef(self.src_y_values, self.dst_y_values)[0, 1]
                result["pearson_y"] = float(corr_y) if not np.isnan(corr_y) else None
            except (ValueError, FloatingPointError):
                result["pearson_y"] = None
        else:
            result["pearson_y"] = None

        # Scale ratio for X and Y (from paper: mean absolute value ratio)
        if self.src_sequences and self.dst_sequences:
            src_x = [abs(seq[0]) for seq in self.src_sequences]
            src_y = [abs(seq[1]) for seq in self.src_sequences]
            dst_x = [abs(seq[0]) for seq in self.dst_sequences]
            dst_y = [abs(seq[1]) for seq in self.dst_sequences]

            src_x_mean = np.mean(src_x)
            dst_x_mean = np.mean(dst_x)
            src_y_mean = np.mean(src_y)
            dst_y_mean = np.mean(dst_y)

            # Scale ratio: ensure >= 1 (invert if < 1)
            if dst_x_mean > 0:
                scale_x = src_x_mean / dst_x_mean
                result["scale_ratio_x"] = float(scale_x if scale_x >= 1 else 1 / scale_x)
            else:
                result["scale_ratio_x"] = None

            if dst_y_mean > 0:
                scale_y = src_y_mean / dst_y_mean
                result["scale_ratio_y"] = float(scale_y if scale_y >= 1 else 1 / scale_y)
            else:
                result["scale_ratio_y"] = None
        else:
            result["scale_ratio_x"] = None
            result["scale_ratio_y"] = None

        result["sample_count"] = len(self.src_x_values)
        return result


class MouseButtonMetric:
    """Mouse button metrics: per-button count accuracy."""

    name = "mouse_button"

    def __init__(self, cfg: BinningConfig):
        self.empty_bins_as_correct = cfg.empty_bins_as_correct
        self.button_accuracies: List[float] = []

    def update(self, src_bin: Bin, dst_bin: Bin):
        """Update the metric with a new bin."""
        src_counts = self._count_button_events(src_bin)
        dst_counts = self._count_button_events(dst_bin)

        all_buttons = set(src_counts.keys()) | set(dst_counts.keys())
        if not all_buttons:
            if self.empty_bins_as_correct:
                self.button_accuracies.append(1.0)
            return

        for button in all_buttons:
            src_count = src_counts.get(button, 0)
            dst_count = dst_counts.get(button, 0)
            accuracy = 1.0 if src_count == dst_count else 0.0
            self.button_accuracies.append(accuracy)

    def _count_button_events(self, bin_data: Bin) -> Dict[str, int]:
        """Count button events by type in a bin."""
        counts = defaultdict(int)
        for event in bin_data["events"]:
            if _determine_event_type(event) == EventType.MOUSE_OP:
                decoded = event.decoded
                flags = decoded.button_flags
                if flags & RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_DOWN:
                    counts["left_down"] += 1
                if flags & RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_UP:
                    counts["left_up"] += 1
                if flags & RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_DOWN:
                    counts["right_down"] += 1
                if flags & RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_UP:
                    counts["right_up"] += 1
                if flags & RawMouseEvent.ButtonFlags.RI_MOUSE_MIDDLE_BUTTON_DOWN:
                    counts["middle_down"] += 1
                if flags & RawMouseEvent.ButtonFlags.RI_MOUSE_MIDDLE_BUTTON_UP:
                    counts["middle_up"] += 1
        return dict(counts)

    @property
    def metric(self) -> Dict[str, Any]:
        """Return the current metric value."""
        if self.button_accuracies:
            return {
                "button_accuracy": float(np.mean(self.button_accuracies)),
                "sample_count": len(self.button_accuracies),
            }
        return {"button_accuracy": None, "sample_count": 0}


class KeyboardMetric:
    """Keyboard metrics: per-key count accuracy."""

    name = "keyboard"

    def __init__(self, cfg: BinningConfig):
        self.empty_bins_as_correct = cfg.empty_bins_as_correct
        self.key_accuracies: List[float] = []

    def update(self, src_bin: Bin, dst_bin: Bin):
        """Update the metric with a new bin."""
        src_counts = self._count_key_events(src_bin)
        dst_counts = self._count_key_events(dst_bin)

        all_keys = set(src_counts.keys()) | set(dst_counts.keys())
        if not all_keys:
            if self.empty_bins_as_correct:
                self.key_accuracies.append(1.0)
            return

        for key in all_keys:
            src_count = src_counts.get(key, 0)
            dst_count = dst_counts.get(key, 0)
            accuracy = 1.0 if src_count == dst_count else 0.0
            self.key_accuracies.append(accuracy)

    def _count_key_events(self, bin_data: Bin) -> Dict[str, int]:
        """Count keyboard events by key and type in a bin."""
        counts = defaultdict(int)
        for event in bin_data["events"]:
            if _determine_event_type(event) == EventType.KEYBOARD:
                decoded = event.decoded
                if hasattr(decoded, "event_type") and hasattr(decoded, "vk"):
                    key = f"{decoded.event_type}_{decoded.vk}"
                    counts[key] += 1
        return dict(counts)

    @property
    def metric(self) -> Dict[str, Any]:
        """Return the current metric value."""
        if self.key_accuracies:
            return {
                "key_accuracy": float(np.mean(self.key_accuracies)),
                "sample_count": len(self.key_accuracies),
            }
        return {"key_accuracy": None, "sample_count": 0}


def _get_first_screen_info(reader: OWAMcapReader) -> Tuple[int, int]:
    """Get (mcap_timestamp, pts_ns) of the first screen event."""
    for msg in reader.iter_messages(topics=["screen"]):
        pts_ns = msg.decoded.media_ref.pts_ns
        return msg.timestamp, pts_ns
    raise ValueError("No screen events found in MCAP file")


def _normalize_events(reader: OWAMcapReader, offset: int, start_pts: int, end_pts: int) -> Iterator[McapMessage]:
    """Normalize event timestamps to pts_ns-based time and filter by range."""
    for event in reader.iter_messages(topics=["screen", "keyboard", "mouse/raw"]):
        normalized_ts = event.timestamp - offset
        if start_pts <= normalized_ts <= end_pts:
            event.timestamp = normalized_ts
            yield event


def compute_binned_metrics(src_mcap_path: str, dst_mcap_path: str, cfg: BinningConfig) -> Dict[str, Any]:
    """
    Compute binned metrics from two MCAP files.

    Timestamps are aligned based on the first screen event's pts_ns in each file.
    This allows comparing mcap files with different absolute timestamps (e.g., 2026 vs 1970).

    Args:
        src_mcap_path: Path to source/ground truth MCAP file
        dst_mcap_path: Path to predicted/destination MCAP file
        cfg: Binning configuration

    Returns:
        Dictionary of computed metrics
    """
    with OWAMcapReader(src_mcap_path) as src_reader, OWAMcapReader(dst_mcap_path) as dst_reader:
        # Get first screen event info for alignment
        src_mcap_ts, src_pts = _get_first_screen_info(src_reader)
        dst_mcap_ts, dst_pts = _get_first_screen_info(dst_reader)

        # Calculate offsets: normalized_ts = mcap_ts - offset = pts_ns-based time
        src_offset = src_mcap_ts - src_pts
        dst_offset = dst_mcap_ts - dst_pts

        # Determine common time range in pts_ns space
        # src duration in pts_ns: from src_pts to src_pts + (src_reader.end_time - src_mcap_ts)
        src_end_pts = src_pts + (src_reader.end_time - src_mcap_ts)
        dst_end_pts = dst_pts + (dst_reader.end_time - dst_mcap_ts)

        start_pts = max(src_pts, dst_pts)
        end_pts = min(src_end_pts, dst_end_pts)
        duration_ns = end_pts - start_pts
        duration_sec = duration_ns / 1e9

        logger.info(f"Source: first_screen_ts={src_mcap_ts}, first_pts={src_pts}, offset={src_offset}")
        logger.info(f"Dest:   first_screen_ts={dst_mcap_ts}, first_pts={dst_pts}, offset={dst_offset}")
        logger.info(f"Common pts range: {start_pts} - {end_pts} (duration: {duration_sec:.2f}s)")

        # Handle no-overlap case
        if end_pts <= start_pts:
            raise ValueError(
                f"No temporal overlap between source and destination MCAP files. "
                f"Source pts range: [{src_pts}, {src_end_pts}], "
                f"Destination pts range: [{dst_pts}, {dst_end_pts}]"
            )

        # Create bins in normalized (pts_ns-based) time
        bins = create_bins(start_pts, end_pts, cfg.bin_size_ns)
        logger.info(f"Created {len(bins)} bins of size {cfg.bin_size_ns / 1e6:.1f}ms")

        # Get normalized events from both files
        src_events = _normalize_events(src_reader, src_offset, start_pts, end_pts)
        dst_events = _normalize_events(dst_reader, dst_offset, start_pts, end_pts)

        # Bin the events
        src_bins = bin_events(src_events, copy.deepcopy(bins))
        dst_bins = bin_events(dst_events, copy.deepcopy(bins))

        # Initialize metrics
        metrics = [
            MouseMoveMetric(cfg),
            MouseButtonMetric(cfg),
            KeyboardMetric(cfg),
        ]

        # Process each bin pair
        for src_bin, dst_bin in zip(src_bins, dst_bins):
            for m in metrics:
                m.update(src_bin, dst_bin)

        # Aggregate results
        results: Dict[str, Any] = {
            "config": {
                "bin_size_ms": cfg.bin_size_ns / 1e6,
                "empty_bins_as_correct": cfg.empty_bins_as_correct,
            },
            "summary": {
                "duration_sec": duration_sec,
                "num_bins": len(bins),
            },
        }

        for m in metrics:
            metric_results = m.metric
            results[m.name] = metric_results

        return results


def main():
    parser = argparse.ArgumentParser(
        description="G-IDM evaluation: compare ground truth MCAP with predicted MCAP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run evaluate.py ground_truth.mcap predicted.mcap
    uv run evaluate.py gt.mcap pred.mcap --bin-ms 100
    uv run evaluate.py gt.mcap pred.mcap --output results.json --verbose
""",
    )
    parser.add_argument("ground_truth", type=str, help="Path to ground truth MCAP file")
    parser.add_argument("predicted", type=str, help="Path to predicted MCAP file")
    parser.add_argument("--bin-ms", type=int, default=50, help="Bin size in milliseconds (default: 50)")
    parser.add_argument(
        "--empty-bins-as-correct",
        action="store_true",
        help="Treat empty bins as correct matches",
    )
    parser.add_argument("--output", "-o", type=str, help="Save JSON output to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    gt_path = Path(args.ground_truth)
    pred_path = Path(args.predicted)

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth MCAP not found: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Predicted MCAP not found: {pred_path}")

    cfg = BinningConfig(
        bin_size_ns=args.bin_ms * 1_000_000,
        empty_bins_as_correct=args.empty_bins_as_correct,
    )

    logger.info(f"Evaluating: {gt_path} vs {pred_path}")
    results = compute_binned_metrics(str(gt_path), str(pred_path), cfg)

    # Print results
    output = json.dumps(results, indent=2)
    print(output)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

    logger.success("Evaluation complete")


if __name__ == "__main__":
    main()
