# Pyramid Sparse Attention (PSA)

[**English**](README.md) | [**中文**](README_zh.md)

**Website:** [http://ziplab.co/PSA](http://ziplab.co/PSA)

Training-free inference acceleration for video generation models.

## Installation

### Using uv (Recommended)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> For best performance, we recommend using PyTorch nightly version.

## Download Weights

### CogVideoX-5B LoRA (4-step)

```bash
huggingface-cli download GYP666/BLADE cogvideox-5b-psa-lora/pytorch_lora_weights.safetensors --local-dir ./weights
```

**Note:** After downloading, update the `lora_path` in `examples/configs/model_configs.py` to point to your weights directory.

## Quick Start

### CogVideoX1.5-5B

```bash
python examples/inference/cogvideo/cogvideo_5b.py \
    --model cogvideo1.5_5b \
    --prompt "your prompt here" \
    --use_psa
```

### Wan2.1-1.3B

```bash
python examples/inference/wan21/wan21_1.3b.py \
    --prompt "your prompt here" \
    --use_psa --no_warmup
```

For more inference examples, see [examples/INFERENCE.md](examples/INFERENCE.md).

## Attention Configuration

The PSA behavior is configured via `configs/attention_config.yaml`. Each model has its own configuration section.

### Configuration Structure

```yaml
ModelName:
  default_attention: psa_balanced    # Default preset to use
  video_scale:                       # Video dimension divisors
    width_divisor: 16
    height_divisor: 16
    depth_divisor: 4
  text_length: 226                   # Text token length (model-specific)
  attention_configs:
    preset_name:                     # e.g., psa_balanced, psa_4steps, baseline
      type: psa                      # "psa" for sparse attention, "dense" for baseline
      description: "..."
      # PSA-specific parameters below
```

### Key Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `type` | Attention type | `psa` (sparse) or `dense` (baseline) |
| `use_rearrange` | Enable spatial rearrangement | `true` / `false` |
| `block_size.m` | Query block size | `128` |
| `block_size.n` | Key/Value block size | `32`, `128` |
| `block_size.tile_n` | Tile size for K/V | `32` |
| `mask_ratios` | Sparsity ratio per pyramid level | See below |
| `mask_mode` | Mask selection mode | `thresholdbound`, `topk` |
| `warmup_steps` | Dense attention warmup steps | `0`, `12`, `15` |
| `rearrange_method` | Token rearrangement algorithm | `Gilbert` |

### Mask Ratios Explained

The `mask_ratios` parameter defines sparsity levels for each pyramid level:

```yaml
mask_ratios:
  1: [0.0, 0.4]    # Level 1: timesteps 0%-40% use full attention
  2: [0.4, 0.5]    # Level 2: timesteps 40%-50% use 2x downsampled attention
  4: [0.5, 0.6]    # Level 4: timesteps 50%-60% use 4x downsampled attention
  8: [0.6, 0.8]    # Level 8: timesteps 60%-80% use 8x downsampled attention
  0: [0.8, 1.0]    # Level 0: timesteps 80%-100% skip attention (most sparse)
```

- **Level 1**: Full resolution attention (highest quality, slowest)
- **Level 2/4/8**: Progressively downsampled attention (faster with some quality trade-off)
- **Level 0**: Attention skipped entirely (fastest, for final denoising steps)

### Available Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `baseline` | Dense attention without sparsity | Quality baseline, slowest |
| `psa_balanced` | Balanced speed/quality | General use (50 steps) |
| `psa_4steps` | Optimized for 4-step LoRA | Fast inference with LoRA |

### Customizing Configuration

1. Edit `configs/attention_config.yaml`
2. Add a new preset under the target model's `attention_configs`
3. Use it via `--attention_preset your_preset_name`

Example custom preset:
```yaml
CogVideo_5b:
  attention_configs:
    my_custom_preset:
      type: psa
      description: "My custom PSA configuration"
      use_rearrange: true
      block_size:
        m: 128
        n: 64
        tile_n: 32
      mask_ratios:
        1: [0.0, 0.5]
        2: [0.5, 0.7]
        4: [0.7, 0.9]
        0: [0.9, 1.0]
      mask_mode: thresholdbound
      warmup_steps: 10
```
