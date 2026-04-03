# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a TU Delft MSc thesis on **fast super-resolution fluorescence microscopy**. The goal is to improve inference latency of deep-learning SR methods to make them a real-time alternative to SOFI (Super-Resolution Optical Fluctuation Imaging). Two model families are studied:

- **TR-MISR** (Transformer-based): adapts the PROBA-V satellite SR framework to fuse N low-resolution SOFI frames into a single HR image via multi-head self-attention with a learnable class token.
- **MISRGRU** (GRU-based): sequential frame fusion baseline, also with a streaming variant (N-in → N-out).

Both produce **2n−7 upscaling**: 64×64 LR patches → 121×121 HR patches.

The codebase lives at `../TR-MISR/` relative to this latex folder.

---

## Thesis (LaTeX)

### Build Commands

```bash
# Compile (XeLaTeX required for TU Delft fonts; LuaLaTeX also works)
xelatex report.tex
biber report
xelatex report.tex
xelatex report.tex

# Or with latexmk (recommended)
latexmk -xelatex report.tex

# Clean auxiliary files
latexmk -c
```

### Structure

- `report.tex` — main file; all chapters are `\input{...}` here
- `frontmatter/` — title page, preface, summary, nomenclature
- `mainmatter/` — introduction, background, methods, results, discussion, conclusion
  - `sofi_trmisr.tex` — primary research chapter
  - `background/fluorescence_microscopy.tex` — domain background
- `appendix/` — supplementary material
- `report.bib` — BibLaTeX bibliography (backend: biber)
- `tudelft-report.cls` — document class (based on `book`; don't modify unless necessary)
- `figures/` — thesis figures; TR-MISR experiment figures are at `../TR-MISR/figures/`

---

## TR-MISR Codebase (`../TR-MISR/`)

### Setup

```bash
conda create -n trmisr python=3.11
conda activate trmisr
pip install -r requirements.txt   # includes CUDA-compiled torch/torchvision
```

### Common Commands

```bash
# Train TR-MISR
python src/train.py --config config/config_pt.json 2>&1 | tee training.log

# Train streaming MISRGRU
python src/train_streaming_chunked.py --config config/config_streaming.json

# Monitor training
tensorboard --logdir=tb_logs_pt/

# Compare multiple trained models (edit model paths in script first)
python src/compare_models_multi.py

# Evaluate streaming model at different frame counts [1,5,10,20,40,80]
python src/eval_streaming.py

# Benchmark inference speed / GPU memory
python src/scripts/benchmark_inference.py

# Interactive streaming visualization
python src/stream_viewer.py --input /path/to/input.pt --weights /path/to/weights.pth
```

### HPC (DelftBlue)

```bash
cd apptainer/
sbatch build_container.sh      # build Apptainer container (SLURM)
sbatch run_training.sh         # run training inside container
```
Bind-mounts data at `/data`; update `paths.microscopy_data_dir` in config accordingly.

---

## Architecture Overview

### Three-Stage Pipeline (TRNet)

1. **Encoder** (`src/DeepNetworks/TRNet.py`) — residual blocks with PReLU; each LR frame encoded independently → `(B, T, C, H/2, W/2)`
2. **Transformer Fusion** — spatial features flattened to sequence; learnable class token aggregates cross-frame info via multi-head self-attention; alpha masks zero out padded frames
3. **Decoder** — `ConvTranspose2d` layers to reach exact 2n−7 output size (SOFI-specific; replaces original PixelShuffle)

### GRU Baseline (MISRGRU)

- `src/DeepNetworks/MISRGRU.py` — standard sequential GRU fusion; hidden state carries accumulated context
- `src/DeepNetworks/MISRGRU_stream.py` — streaming variant; processes frames one-by-one, emits one SR output per frame (N-in → N-out)
- `src/DeepNetworks/MISRGRU_kbuf.py` — circular buffer variant

### Data Flow

```
microscopy_data/Pt_train/training_tensor_{i}.pt   → (100, 64, 64) LR frames
microscopy_data/Pt_train/target_tensor_{i}.pt     → (121, 121)    HR target
```
`LazyTensorDataset` loads `.pt` files on demand. Collation selects `min_L` frames, pads shorter sequences with alpha=0, and adds a channel dimension:
```
lrs:    (B, min_L, 1, 64, 64)
alphas: (B, min_L)              # 1=real frame, 0=padding
hrs:    (B, 1, 121, 121)
```

### Configuration System

All hyperparameters live in `config/*.json`. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `training.min_L` | Number of LR frames to fuse |
| `training.loss_depend` | `"L1"`, `"Fourier"`, `"L1_Fourier"`, `"Composite"` |
| `training.strategy` | LR scheduler: `0`=ReduceLROnPlateau, `1`=CosineAnnealing, `2`=Manual |
| `network.encoder.channel_size` | Feature map width |
| `network.transformer.depth` / `.heads` | Transformer depth and attention heads |

The optimizer uses **two LR groups**: `lr_coder` (encoder/decoder) and `lr_transformer` (typically lower).

### Key Implementation Notes

- **`TRNet.py` forward pass** has a `print(preds.shape)` debug statement — remove before production training runs.
- Checkpoints saved to `models/weights_pt/{folder}/{model_name}.pth`; periodic saves every 20 epochs as `TRNet{epoch}.pth`.
- All guides and experiment summaries go in `docs/`.
