# PSA Attention

Plug-and-play Pyramid Sparse Attention module.

## Usage

### Default Config

```python
from psa_triton import PSAAttention

psa = PSAAttention()
out = psa(q, k, v)  # q, k, v: [B, H, L, D]
```

### Custom Config

```python
from psa_triton import PSAAttention, PSAConfig

# Only override what you need, rest uses defaults
config = PSAConfig(mask_mode='thresholdbound')
psa = PSAAttention(config)
```

### Full Config Reference

```python
config = PSAConfig(
    # Block size configuration
    block_m=128,          # Query block size
    block_n=64,           # Key/Value block size
    tile_n=32,            # Tile size for K/V processing
    
    # Mask ratio configuration
    mask_ratios={
        1: (0.0, 0.1),    # 10% full attention
        2: (0.1, 0.15),   # 5% with 2x pooling
        4: (0.15, 0.15),  # 0% with 4x pooling
        8: (0.15, 0.35),  # 20% with 8x pooling
        0: (0.35, 1.0),   # 65% skipped
    },
    mask_mode='topk',     # 'topk' or 'thresholdbound'
    
    # Similarity-based pooling constraint
    use_sim_mask=False,
    sim_2x_threshold=0.0,
    sim_4x_threshold=0.0,
    sim_8x_threshold=-1.0,
)
psa = PSAAttention(config)
out = psa(q, k, v)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_m` | int | 128 | Query block size |
| `block_n` | int | 64 | Key/Value block size |
| `tile_n` | int | 32 | Tile size for K/V processing |
| `mask_ratios` | dict | See above | Sparsity distribution per pyramid level |
| `mask_mode` | str | `'topk'` | `'topk'` (fixed quota) or `'thresholdbound'` (dynamic) |
| `use_sim_mask` | bool | False | Enable similarity-based pooling constraint (requires `attn_impl='old_mask_type'`) |
| `sim_2x_threshold` | float | 0.0 | Similarity threshold for 2x pooling |
| `sim_4x_threshold` | float | 0.0 | Similarity threshold for 4x pooling |
| `sim_8x_threshold` | float | -1.0 | Similarity threshold for 8x pooling |
| `rearrange_method` | str | None | Token rearrangement method: `'Gilbert'`, `'STA'`, `'SemanticAware'`, `'Hybrid'` |
| `attn_impl` | str | `'new_mask_type'` | Kernel implementation: `'new_mask_type'` (default) or `'old_mask_type'` |

> **Note**:
> - `use_sim_mask=True` requires `attn_impl='old_mask_type'`. The similarity-based pooling constraint is not compatible with `new_mask_type` due to mask format differences.
> - `attn_impl='old_mask_type'` only supports `block_m=128` and `block_n=128`. Use `new_mask_type` for other block sizes (128, 64, 32).
