# PSA Attention

即插即用的 Pyramid Sparse Attention 模块。

## 使用方法

### 默认配置

```python
from psa_triton import PSAAttention

psa = PSAAttention()
out = psa(q, k, v)  # q, k, v: [B, H, L, D]
```

### 自定义配置

```python
from psa_triton import PSAAttention, PSAConfig

# 只修改需要的参数，其余使用默认值
config = PSAConfig(mask_mode='thresholdbound')
psa = PSAAttention(config)
```

### 完整配置参考

```python
config = PSAConfig(
    # Block 大小配置
    block_m=128,          # Query block 大小
    block_n=64,           # Key/Value block 大小
    tile_n=32,            # K/V 处理的 tile 大小
    
    # Mask ratio 配置
    mask_ratios={
        1: (0.0, 0.1),    # 10% full attention
        2: (0.1, 0.15),   # 5% with 2x pooling
        4: (0.15, 0.15),  # 0% with 4x pooling
        8: (0.15, 0.35),  # 20% with 8x pooling
        0: (0.35, 1.0),   # 65% skipped
    },
    mask_mode='topk',     # 'topk' 或 'thresholdbound'
    
    # 相似度约束的 pooling 配置
    use_sim_mask=False,
    sim_2x_threshold=0.0,
    sim_4x_threshold=0.0,
    sim_8x_threshold=-1.0,
)
psa = PSAAttention(config)
out = psa(q, k, v)
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `block_m` | int | 128 | Query block 大小 |
| `block_n` | int | 64 | Key/Value block 大小 |
| `tile_n` | int | 32 | K/V 处理的 tile 大小 |
| `mask_ratios` | dict | 见上 | 各 pyramid level 的稀疏度分布 |
| `mask_mode` | str | `'topk'` | `'topk'` (固定配额) 或 `'thresholdbound'` (动态分配) |
| `use_sim_mask` | bool | False | 启用相似度约束的 pooling |
| `sim_2x_threshold` | float | 0.0 | 2x pooling 的相似度阈值 |
| `sim_4x_threshold` | float | 0.0 | 4x pooling 的相似度阈值 |
| `sim_8x_threshold` | float | -1.0 | 8x pooling 的相似度阈值 |
