# 金字塔稀疏注意力 (PSA)

[**English**](README.md) | [**中文**](README_zh.md)

**项目主页:** [http://ziplab.co/PSA](http://ziplab.co/PSA)

无需训练的视频生成模型推理加速方案。

## 安装

### 使用 uv (推荐)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 使用 pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> 为获得最佳性能，建议使用 PyTorch nightly 版本。

## 下载权重

### CogVideoX-5B LoRA (4步推理)

```bash
huggingface-cli download GYP666/BLADE cogvideox-5b-psa-lora/pytorch_lora_weights.safetensors --local-dir ./weights
```

**注意：** 下载后需要修改 `examples/configs/model_configs.py` 中的 `lora_path` 指向你的权重目录。

## 快速开始

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

更多推理示例请参考 [examples/INFERENCE_zh.md](examples/INFERENCE_zh.md)。

## 注意力配置说明

PSA 的行为通过 `configs/attention_config.yaml` 文件配置。每个模型都有独立的配置节。

### 配置文件结构

```yaml
ModelName:                           # 模型名称
  default_attention: psa_balanced    # 默认使用的预设
  video_scale:                       # 视频维度除数
    width_divisor: 16
    height_divisor: 16
    depth_divisor: 4
  text_length: 226                   # 文本token长度（模型相关）
  attention_configs:
    preset_name:                     # 预设名称：psa_balanced, psa_4steps, baseline
      type: psa                      # "psa" 稀疏注意力，"dense" 基线
      description: "..."
      # PSA 特定参数见下文
```

### 核心参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `type` | 注意力类型 | `psa`（稀疏）或 `dense`（密集基线） |
| `use_rearrange` | 启用空间重排 | `true` / `false` |
| `block_size.m` | Query 块大小 | `128` |
| `block_size.n` | Key/Value 块大小 | `32`, `128` |
| `block_size.tile_n` | K/V 的 Tile 大小 | `32` |
| `mask_ratios` | 每个金字塔层级的稀疏比例 | 见下文 |
| `mask_mode` | 掩码选择模式 | `thresholdbound`, `topk` |
| `warmup_steps` | 密集注意力预热步数 | `0`, `12`, `15` |
| `rearrange_method` | Token 重排算法 | `Gilbert` |

### mask_ratios 参数详解

`mask_ratios` 定义了每个金字塔层级的稀疏程度：

```yaml
mask_ratios:
  1: [0.0, 0.4]    # 层级1：时间步0%-40%使用完整注意力
  2: [0.4, 0.5]    # 层级2：时间步40%-50%使用2倍下采样注意力
  4: [0.5, 0.6]    # 层级4：时间步50%-60%使用4倍下采样注意力
  8: [0.6, 0.8]    # 层级8：时间步60%-80%使用8倍下采样注意力
  0: [0.8, 1.0]    # 层级0：时间步80%-100%跳过注意力（最稀疏）
```

- **层级 1**：全分辨率注意力（最高质量，最慢）
- **层级 2/4/8**：逐级下采样注意力（更快，质量略有损失）
- **层级 0**：完全跳过注意力（最快，用于最终去噪步骤）

### 可用预设

| 预设 | 说明 | 使用场景 |
|------|------|----------|
| `baseline` | 无稀疏的密集注意力 | 质量基线，最慢 |
| `psa_balanced` | 速度与质量平衡 | 通用场景（50步） |
| `psa_4steps` | 针对4步LoRA优化 | 使用LoRA的快速推理 |

### 自定义配置

1. 编辑 `configs/attention_config.yaml`
2. 在目标模型的 `attention_configs` 下添加新预设
3. 通过 `--attention_preset your_preset_name` 使用

自定义预设示例：
```yaml
CogVideo_5b:
  attention_configs:
    my_custom_preset:
      type: psa
      description: "我的自定义PSA配置"
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
