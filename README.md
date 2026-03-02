# D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI

Suhwan Choi*, Jaeyoon Jung*, Haebin Seong*, Minchan Kim, Minyeong Kim, Yongjun Cho, Yoonshik Kim, Yubeen Park, Youngjae Yu‡, Yunsung Lee‡

[![Project Page](https://img.shields.io/badge/🌐_Project_Page-blue?style=flat-square)](https://worv-ai.github.io/d2e/)
[![arXiv](https://img.shields.io/badge/📄_arXiv-2510.05684-red?style=flat-square)](https://arxiv.org/abs/2510.05684)
[![Demo](https://img.shields.io/badge/🤗_Demo-Generalist--IDM-yellow?style=flat-square)](https://huggingface.co/spaces/lastdefiance20/Generalist-IDM)
[![Model](https://img.shields.io/badge/🤗_Model-1B-yellow?style=flat-square)](https://huggingface.co/open-world-agents/Generalist-IDM-1B)
[![Dataset 480p](https://img.shields.io/badge/🤗_Dataset-480p-yellow?style=flat-square)](https://huggingface.co/datasets/open-world-agents/D2E-480p)
[![Dataset Original](https://img.shields.io/badge/🤗_Dataset-FHD/QHD-yellow?style=flat-square)](https://huggingface.co/datasets/open-world-agents/D2E-Original)

<img width="5334" height="2306" alt="image" src="https://github.com/user-attachments/assets/5a6dd6a1-201f-4da6-8530-340b5edfa2b1" />

## News

- [2026/03/02] We release the evaluation script [`evaluate.py`](./evaluate.py). Existing desktop IDM works each define their own ad-hoc metrics, making fair comparison difficult. We provide a standardized evaluation protocol with well-defined, reproducible metrics (see [Evaluation](#evaluation)).

- [2026/01/15] We release the Generalist-IDM demo on Hugging Face Spaces [lastdefiance20/Generalist-IDM](https://huggingface.co/spaces/lastdefiance20/Generalist-IDM), the model weights [open-world-agents/Generalist-IDM-1B](https://huggingface.co/open-world-agents/Generalist-IDM-1B), and the inference code [`inference.py`](./inference.py).

- [2025/12/18] We release the FHD/QHD versions of the dataset on Hugging Face [open-world-agents/D2E-Original](https://huggingface.co/datasets/open-world-agents/D2E-Original) for training world models and video generation models. We also fix issues in the 480p dataset [open-world-agents/D2E-480p](https://huggingface.co/datasets/open-world-agents/D2E-480p).

- [2025/12/01] We release 480p version of dataset at huggingface. [open-world-agents/D2E-480p](https://huggingface.co/datasets/open-world-agents/D2E-480p): **267 hours** of synchronized video, audio, and input events from **29** PC games across diverse genres (FPS, open-world, sandbox, and more), for training vision-action models and game agents.

- [2025/10/21] We release part of our source codes. Code is comming soon! `ocap` and `owa` toolkit is being open-sourced already, have a look at these first.
  - https://github.com/open-world-agents/ocap: ocap (Omnimodal CAPture) captures all essential desktop signals in synchronized format. Records screen video, audio, keyboard/mouse input, and window events.
  - https://github.com/open-world-agents/open-world-agents: A versatile and efficient monorepo that embraces and grows multiple projects, containing all the essential building blocks for agent development.
  - https://worv-ai.github.io/d2e/: D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI. Code will coming soon!

## Dataset

We provide **267 hours** of synchronized video, audio, and input events from **29 PC games** across diverse genres (FPS, open-world, sandbox, and more).

| Dataset | Resolution | Use Case |
|---------|------------|----------|
| [open-world-agents/D2E-480p](https://huggingface.co/datasets/open-world-agents/D2E-480p) | 480p 60fps | Vision-action model training |
| [open-world-agents/D2E-Original](https://huggingface.co/datasets/open-world-agents/D2E-Original) | FHD/QHD | World models, video generation |

### What's Included

- **Video + Audio**: H.264 encoded at 60fps with game audio
- **Input Events**: Keyboard (press/release), mouse (clicks, coordinates, raw HID deltas)—all with nanosecond timestamps
- **OWAMcap Format**: Built on [MCAP](https://mcap.dev/), indexed for fast random access

### Quick Start

```bash
pip install mcap-owa-support owa-msgs huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
from mcap_owa.highlevel import OWAMcapReader

# Download a sample recording
mcap_file = hf_hub_download(
    repo_id="open-world-agents/D2E-480p",
    filename="Apex_Legends/0805_01.mcap",
    repo_type="dataset"
)

with OWAMcapReader(mcap_file) as reader:
    # Load a video frame
    for msg in reader.iter_messages(topics=["screen"]):
        screen = msg.decoded
        screen.resolve_relative_path(mcap_file)
        frame = screen.load_frame_array()  # numpy array (H, W, 3)
        break

    # Read keyboard events
    for msg in reader.iter_messages(topics=["keyboard"]):
        print(msg.decoded)  # KeyboardEvent(event_type='press', vk=87)
        break

    # Read raw mouse events
    for msg in reader.iter_messages(topics=["mouse/raw"]):
        print(msg.decoded)  # RawMouseEvent(last_x=12, last_y=-3, button_flags=0)
        break
```

### Visualize

Explore recordings in your browser with synchronized keyboard/mouse overlay: [Open in Dataset Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset)

<img src="https://github.com/open-world-agents/owa-dataset-visualizer/blob/main/.github/assets/viewer.png?raw=true" alt="Dataset Visualizer Preview" width="600">

### For Training

We provide [`owa-data`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-data), a data pipeline that converts this dataset into HuggingFace Datasets ready for PyTorch DataLoader with tokenization and sequence packing.

## Inference

Run action prediction on any gameplay video using [`inference.py`](./inference.py). The script uses [uv](https://docs.astral.sh/uv/) for dependency management—no manual installation required.

### Prerequisites

- [uv](https://docs.astral.sh/uv/)
- FFmpeg (for video preprocessing)
- CUDA-capable GPU (recommended, ~8GB+ VRAM)

> ⏱️ **Inference Time**: On H100, processing 1 second of video takes ~6 seconds. For a 1-minute video, expect ~6 minutes of inference time. Use `--max-duration` to limit video length for faster testing.

### Quick Start

```bash
# Run inference on a video (dependencies are auto-installed by uv)
uv run inference.py input_video.mp4 output.mcap

# Specify a different model or device
uv run inference.py input_video.mp4 output.mcap --model open-world-agents/Generalist-IDM-1B
uv run inference.py input_video.mp4 output.mcap --device cpu

# Limit video duration for faster testing
uv run inference.py input_video.mp4 output.mcap --max-duration 30
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `open-world-agents/Generalist-IDM-1B` | Model path or Hugging Face ID |
| `--device` | `cuda` | Device to run on (`cuda` or `cpu`) |
| `--max-duration` | None | Max video duration in seconds |
| `--max-context-length` | 2048 | Max context length for the model |
| `--time-shift` | 0.1 | Time shift for actions in seconds |

### Output Format

The output is an [MCAP](https://mcap.dev/) file containing predicted keyboard and mouse events with timestamps synchronized to the input video. You can visualize the output using the [Dataset Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset).

## Evaluation

Existing desktop IDM works(e.g. VPT) each define their own ad-hoc metrics—varying bin sizes, inconsistent aggregation, or metrics that conflate movement direction with magnitude—making cross-paper comparison unreliable. We provide [`evaluate.py`](./evaluate.py), a standardized evaluation protocol with clearly defined metrics (paper Section F.2). All metrics are computed over **non-overlapping 50ms temporal bins** (configurable via `--bin-ms`):

| Category | Metric | Description |
|----------|--------|-------------|
| Mouse movement | **Pearson correlation (X/Y)** | Direction alignment between predicted and ground truth mouse deltas |
| Mouse movement | **Scale ratio (X/Y)** | Magnitude consistency (ratio of mean absolute deltas, always ≥1) |
| Mouse buttons | **Per-button accuracy** | Exact count match per button type (left/right/middle, down/up) per bin |
| Keyboard | **Per-key accuracy** | Exact count match per key event type per bin |

```bash
# Inference → Evaluation
uv run inference.py gameplay.mp4 predicted.mcap --max-duration 30
uv run evaluate.py ground_truth.mcap predicted.mcap --output results.json
```

Output example:

```json
{
  "summary": { "duration_sec": 30.0, "num_bins": 600 },
  "mouse_move": { "pearson_x": 0.82, "pearson_y": 0.78, "scale_ratio_x": 1.15, "scale_ratio_y": 1.22 },
  "mouse_button": { "button_accuracy": 0.91 },
  "keyboard": { "key_accuracy": 0.85 }
}
```

We encourage the community to adopt this protocol for desktop IDM benchmarking—let's make evaluation reproducible.

## Citation

If you find this work useful, please cite our paper:

```
@article{choi2025d2e,
  title={D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI},
  author={Choi, Suhwan and Jung, Jaeyoon and Seong, Haebin and Kim, Minchan and Kim, Minyeong and Cho, Yongjun and Kim, Yoonshik and Park, Yubeen and Yu, Youngjae and Lee, Yunsung},
  journal={arXiv preprint arXiv:2510.05684},
  year={2025}
}
```
