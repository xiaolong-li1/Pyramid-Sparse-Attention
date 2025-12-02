"""
Attention mechanism modification tools for CogVideo and Wan models

Supports PSA (Pyramid Sparse Attention) mechanism

Usage example:
    # PSA attention
    set_adaptive_sparse_attention(
        model=model,
        model_name="Wan2.1_14b",
        inference_num=50,
        video_shape=[1280, 720, 480],
        attention_type="PSA"
    )
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import yaml
import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
from diffusers.models.attention_processor import Attention
from diffusers.models import CogVideoXTransformer3DModel
from psa_triton.pyramid_sparse_attention import PyramidSparseAttention, AttentionConfig
from diffusers.models.embeddings import apply_rotary_emb

sparsity_record = []

# List of supported model names
SUPPORTED_MODELS = {
    "CogVideo": ["CogVideo_5b", "CogVideo_2b"],
    "Wan": ["Wan2.1_14b", "Wan2.1_1.3b", "Wan2.1_1.3b_4steps", "Wan2.2_A14B", "Wan2.2_5B"]
}


class WanAttentionConfigLoader:
    """Attention configuration loader for Wan models, supports multiple presets"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "attention_config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

    def list_presets(self, model_name: str) -> List[str]:
        model_cfg = self._config.get(model_name)
        if model_cfg is None:
            raise ValueError(
                f"Model '{model_name}' not found in config. "
                f"Available models: {list(self._config.keys())}"
            )
        presets = model_cfg.get("attention_configs", {})
        return list(presets.keys())

    def _normalise_video_shape(self, video_shape: Union[List[int], Tuple[int, int, int]]) -> List[int]:
        if video_shape is None or len(video_shape) != 3:
            raise ValueError("video_shape must be a sequence of [width, height, depth].")
        return [int(video_shape[0]), int(video_shape[1]), int(video_shape[2])]

    def _compute_video_dims(self, model_cfg: Dict[str, Any], video_shape: List[int]) -> Dict[str, int]:
        if "video_scale" not in model_cfg:
            raise ValueError("Missing 'video_scale' configuration for model.")

        scale = model_cfg["video_scale"]
        return {
            "width": math.ceil(video_shape[0] / scale["width_divisor"]),
            "height": math.ceil(video_shape[1] / scale["height_divisor"]),
            "depth": math.ceil(video_shape[2] / scale["depth_divisor"]),
        }

    def get_attention_config(
        self,
        model_name: str,
        preset_name: Optional[str],
        video_shape: Union[List[int], Tuple[int, int, int]],
    ) -> Dict[str, Any]:
        model_cfg = self._config.get(model_name)
        if model_cfg is None:
            raise ValueError(
                f"Model '{model_name}' not found in config. "
                f"Available models: {list(self._config.keys())}"
            )

        presets = model_cfg.get("attention_configs", {})
        if not presets:
            raise ValueError(f"No attention presets defined for model '{model_name}'.")

        if preset_name is None:
            preset_name = model_cfg.get("default_attention")

        if preset_name is None:
            raise ValueError(
                f"Preset name must be provided for model '{model_name}'. "
                f"Available presets: {list(presets.keys())}"
            )

        if preset_name not in presets:
            raise ValueError(
                f"Preset '{preset_name}' not found for model '{model_name}'. "
                f"Available presets: {list(presets.keys())}"
            )

        preset_cfg = presets[preset_name]
        preset_type = preset_cfg.get("type", "psa").lower()
        video_shape = self._normalise_video_shape(video_shape)
        video_dims = self._compute_video_dims(model_cfg, video_shape)

        base_info: Dict[str, Any] = {
            "name": preset_name,
            "type": preset_type,
            "description": preset_cfg.get("description"),
            "video_dims": video_dims,
            "original_video_shape": {
                "width": video_shape[0],
                "height": video_shape[1],
                "depth": video_shape[2],
            },
        }

        if preset_type == "dense":
            return base_info

        if preset_type != "psa":
            raise ValueError(f"Unsupported preset type '{preset_type}' for model '{model_name}'.")

        block_cfg = preset_cfg.get("block_size", {})
        mask_ratios_cfg = preset_cfg.get("mask_ratios", {})
        mask_ratios = {int(k): tuple(v) for k, v in mask_ratios_cfg.items()}
        tile_size = preset_cfg.get("tile_size")
        if tile_size is not None:
            tile_size = tuple(tile_size)

        sim_thresholds = preset_cfg.get("sim_thresholds", {})

        params: Dict[str, Any] = {
            "width": video_dims["width"],
            "height": video_dims["height"],
            "depth": video_dims["depth"],
            "text_length": preset_cfg.get("text_length", model_cfg.get("text_length", 0)),
            "use_rearrange": preset_cfg.get("use_rearrange", True),
            "use_sim_mask": preset_cfg.get("use_sim_mask", True),
            "block_m": block_cfg.get("m", 128),
            "block_n": block_cfg.get("n", 128),
            "tile_n": block_cfg.get("tile_n", 32),
            "mask_ratios": mask_ratios,
            "mask_mode": preset_cfg.get("mask_mode", "thresholdbound"),
            "attn_impl": preset_cfg.get("attn_impl", "new_mask_type"),
            "warmup_steps": preset_cfg.get("warmup_steps", 0),
            "tile_size": tile_size,
            "rearrange_method": preset_cfg.get("rearrange_method", "Gilbert"),
            "verbose": preset_cfg.get("verbose", False),
            "enable_logging": preset_cfg.get("enable_logging", True),
            "log_dir": preset_cfg.get("log_dir", "./psa_logs/"),
        }

        if "sim_2x_threshold" in preset_cfg:
            params["sim_2x_threshold"] = preset_cfg["sim_2x_threshold"]
        if "sim_4x_threshold" in preset_cfg:
            params["sim_4x_threshold"] = preset_cfg["sim_4x_threshold"]
        if "sim_8x_threshold" in preset_cfg:
            params["sim_8x_threshold"] = preset_cfg["sim_8x_threshold"]

        if sim_thresholds:
            params.setdefault("sim_2x_threshold", sim_thresholds.get("x2"))
            params.setdefault("sim_4x_threshold", sim_thresholds.get("x4"))
            params.setdefault("sim_8x_threshold", sim_thresholds.get("x8"))

        base_info["params"] = params
        return base_info


# Global config loader instance
_config_loader = None

def get_config_loader(config_path: Optional[str] = None) -> WanAttentionConfigLoader:
    """Get the singleton config loader instance"""
    global _config_loader
    if _config_loader is None or config_path is not None:
        _config_loader = WanAttentionConfigLoader(config_path)
    return _config_loader


class SparseWanAttnProcessor:
    """Attention processor for Wan series models (compatible with Wan2.1 and Wan2.2)"""
    def __init__(self, idx):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )
        self.layer_idx = idx

    def _get_qkv_projections(self, attn, hidden_states, encoder_hidden_states):
        """Get QKV projections - compatible with fused projections (Wan2.2 style)"""
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if getattr(attn, 'fused_projections', False):
            if getattr(attn, 'cross_attention_dim_head', None) is None:
                # In self-attention layers, fuse entire QKV projection
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                # In cross-attention layers, only fuse KV projections
                query = attn.to_q(hidden_states)
                key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        return query, key, value

    def _get_added_kv_projections(self, attn, encoder_hidden_states_img):
        """Get added KV projections for I2V - compatible with fused projections"""
        if getattr(attn, 'fused_projections', False):
            key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
        else:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
        return key_img, value_img

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        numerical_timestep: Optional[torch.Tensor] = None,  # Wan2.2 compatibility
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # Use helper function for QKV projections
        query, key, value = self._get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Wan2.2 style: keep (B, S, H, D) layout instead of (B, H, S, D)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            # Wan2.2 rotary embedding: rotary_emb is a tuple of (freqs_cos, freqs_sin)
            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = self._get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            # Wan2.2 style: keep (B, S, H, D) layout
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            # Use dispatch_attention_fn for consistency with Wan2.2
            from diffusers.models.attention_dispatch import dispatch_attention_fn
            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=None,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # Main attention with inner_attention (sparse attention logic)
        # inner_attention expects (B, H, S, D) layout, need to transpose
        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)

        hidden_states = attn.inner_attention(query_t, key_t, value_t, self.layer_idx)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class SparseCogVideoXAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert attention_mask is None, "Attention mask is not supported"

        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query).to(dtype=value.dtype)
        if attn.norm_k is not None:
            key = attn.norm_k(key).to(dtype=value.dtype)
        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
        hidden_states = attn.inner_attention(query, key, value, self.layer_idx)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states
    

def set_block_sparse_attn_cogvideox(
    model: CogVideoXTransformer3DModel,
    model_name: str,
    inference_num: int,
    attention_info: Dict[str, Any],
    verbose: bool = False,
):
    """Apply specified PSA configuration to CogVideoX models"""
    layer_num = len(model.transformer_blocks)

    # Extract configuration from attention_info
    # Note: get_attention_config returns "video_dims" not "dims"
    video_dims = attention_info.get("video_dims", {})
    original_dims = attention_info.get("original_video_shape", {})
    preset_name = attention_info.get("name")
    description = attention_info.get("description")
    # For CogVideo, params are nested under "params" key (same as Wan)
    params = attention_info.get("params", {})

    print("\n==================== CogVideoX Attention Configuration ====================")
    print(f"Model Name: {model_name}")
    print(f"Attention Preset: {preset_name}")
    if description:
        print(f"Description: {description}")
    print(
        "Original Video Shape: "
        f"width={original_dims.get('width', 'N/A')}, "
        f"height={original_dims.get('height', 'N/A')}, "
        f"depth={original_dims.get('depth', 'N/A')}"
    )
    print(
        "Scaled Video Shape: "
        f"width={video_dims.get('width')}, "
        f"height={video_dims.get('height')}, "
        f"depth={video_dims.get('depth')}"
    )

    config = AttentionConfig(
        width=video_dims.get("width"),
        height=video_dims.get("height"),
        depth=video_dims.get("depth"),
        text_length=params.get("text_length", 226),
        use_rearrange=params.get("use_rearrange", True),
        block_m=params.get("block_m", 128),
        block_n=params.get("block_n", 128),
        tile_n=params.get("tile_n", 32),
        use_sim_mask=params.get("use_sim_mask", False),
        mask_ratios=params.get("mask_ratios", {}),
        mask_mode=params.get("mask_mode", "thresholdbound"),
        attn_impl=params.get("attn_impl", "new_mask_type"),
        tile_size=params.get("tile_size"),
        rearrange_method=params.get("rearrange_method", "Gilbert"),
        warmup_steps=params.get("warmup_steps", 12),
        verbose=verbose,
        sim_2x_threshold=params.get("sim_2x_threshold", 0),
        sim_4x_threshold=params.get("sim_4x_threshold", 0),
        sim_8x_threshold=params.get("sim_8x_threshold", -1),
    )

    print(f"Use Rearrange: {config.use_rearrange}")
    print(f"Rearrange Method: {config.rearrange_method}")
    print(f"Block Size: M={config.block_m}, N={config.block_n}, Tile_N={config.tile_n}")
    if config.tile_size:
        print(f"Tile Size: {config.tile_size}")
    print(f"Mask Mode: {config.mask_mode}")
    print(f"Mask Ratios Configuration:")
    for mask_value, (start_ratio, end_ratio) in config.mask_ratios.items():
        mask_types = {1: "完全注意力", 2: "2x池化", 4: "4x池化", 8: "8x池化", 0: "跳过"}
        interval = end_ratio - start_ratio
        print(f"  - Mask {mask_value} ({mask_types.get(mask_value, 'Unknown')}): {start_ratio:.2f}-{end_ratio:.2f} (interval={interval:.2f})")
    print(f"Sim Mask Enabled: {config.use_sim_mask}")
    print(f"Layer Count: {layer_num}")
    print(f"Inference Steps: {inference_num}")
    print(f"Warmup Steps: {config.warmup_steps}")
    print(f"Verbose Logging: {config.verbose}")
    print(f"=======================================================================\n")

    inner_attn = PyramidSparseAttention(
        config=config,
        inference_num=inference_num,
        layer_num=layer_num,
        model_type="cogvideo",
    )

    for idx, block in enumerate(model.transformer_blocks):
        block.attn1.verbose = verbose
        block.attn1.inner_attention = inner_attn
        origin_processor = block.attn1.get_processor()
        processor = SparseCogVideoXAttnProcessor(idx)
        block.attn1.set_processor(processor)
        if not hasattr(block.attn1, "origin_processor"):
            block.attn1.origin_processor = origin_processor

    return inner_attn


def set_block_sparse_attn_wan(
    model,
    model_name: str,
    inference_num: int,
    attention_info: Dict[str, Any],
    verbose: bool = False,
):
    """Apply specified PSA configuration to Wan series models"""

    layer_num = len(model.blocks)
    params = attention_info["params"]
    config = AttentionConfig(**params)

    video_dims = attention_info.get("video_dims", {})
    original_dims = attention_info.get("original_video_shape", {})
    preset_name = attention_info.get("name")
    description = attention_info.get("description")

    print("\n==================== Wan Attention Configuration ====================")
    print(f"Model Name: {model_name}")
    print(f"Attention Preset: {preset_name}")
    if description:
        print(f"Description: {description}")
    print(
        "Original Video Shape: "
        f"width={original_dims.get('width')}, height={original_dims.get('height')}, depth={original_dims.get('depth')}"
    )
    print(
        "Downscaled Shape: "
        f"width={video_dims.get('width')}, height={video_dims.get('height')}, depth={video_dims.get('depth')}"
    )
    print(f"Text Length: {config.text_length}")
    print(f"Use Rearrange: {config.use_rearrange}")
    print(f"Block Size: M={config.block_m}, N={config.block_n}, Tile_N={config.tile_n}")
    print(f"Tile Size: {config.tile_size}")
    print(f"Rearrange Method: {config.rearrange_method}")
    print(f"Mask Mode: {config.mask_mode}")
    print(f"Attention Implementation: {config.attn_impl}")
    print("Mask Ratios Configuration:")
    for mask_value, (start_ratio, end_ratio) in config.mask_ratios.items():
        mask_type = {1: "Full attention", 2: "2x pooling", 4: "4x pooling", 8: "8x pooling", 0: "Skip"}
        interval = end_ratio - start_ratio
        print(
            f"  - Mask {mask_value} ({mask_type.get(mask_value, 'Unknown')}): "
            f"{start_ratio:.2f}-{end_ratio:.2f} (interval={interval:.2f})"
        )
    print(f"Layer Count: {layer_num}")
    print(f"Inference Steps: {inference_num}")
    print(f"Warmup Steps: {config.warmup_steps}")
    print(f"Verbose Logging: {config.verbose}")
    print("================================================================\n")

    inner_attn = PyramidSparseAttention(
        config=config,
        inference_num=inference_num,
        layer_num=layer_num,
        model_type=model_name,
    )

    for idx, block in enumerate(model.blocks):
        block.attn1.verbose = verbose
        block.attn1.inner_attention = inner_attn
        origin_processor = block.attn1.get_processor()
        processor = SparseWanAttnProcessor(idx)
        block.attn1.set_processor(processor)
        if not hasattr(block.attn1, "origin_processor"):
            block.attn1.origin_processor = origin_processor

    return inner_attn


def set_adaptive_sparse_attention(
    pipe,
    model_name: str,
    inference_num: int = 50,
    video_shape: Optional[Union[List[int], Tuple[int, int, int]]] = None,
    attention_type: Optional[str] = "PSA",
    attention_preset: Optional[str] = None,
    verbose: bool = False,
    config_path: Optional[str] = None,
):
    """Unified attention configuration interface for Wan and CogVideo models, supports multiple presets"""

    if video_shape is None:
        raise ValueError("Please provide video_shape=[width, height, depth] for configuring attention.")

    # Determine model type
    is_cogvideo = "CogVideo" in model_name or "cogvideo" in model_name.lower()
    is_wan = "Wan" in model_name or "wan" in model_name.lower()

    if not is_cogvideo and not is_wan:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models: {list(SUPPORTED_MODELS['CogVideo']) + list(SUPPORTED_MODELS['Wan'])}"
        )

    attn_type_normalized = (attention_type or "psa").lower()
    if attn_type_normalized not in {"psa", "dense", "none"}:
        raise ValueError(f"Unsupported attention_type '{attention_type}'.")

    loader = get_config_loader(config_path)
    attention_info = loader.get_attention_config(model_name, attention_preset, video_shape)

    if attention_info["type"] == "dense" or attn_type_normalized in {"dense", "none"}:
        preset_name = attention_info.get("name")
        reason = "dense attention preset" if attention_info["type"] == "dense" else f"attention_type='{attention_type}'"
        print(
            f"\nℹ️ Skipping sparse attention for {model_name} (reason: {reason}). "
            "No sparse attention will be applied.\n"
        )
        return attention_info

    # Handle CogVideo models
    if is_cogvideo:
        print(f"\n🎥 Detected CogVideo model: {model_name}")
        print("Setting up PSA for CogVideoX transformer...")

        # For CogVideoX, pipe.transformer is the CogVideoXTransformer3DModel
        transformer = pipe.transformer if hasattr(pipe, "transformer") else pipe

        set_block_sparse_attn_cogvideox(
            transformer,
            model_name=model_name,
            inference_num=inference_num,
            attention_info=attention_info,
            verbose=verbose,
        )

        print(f"✅ Successfully configured Pyramid Sparse Attention preset '{attention_info['name']}' for {model_name}\n")
        return attention_info

    # Handle Wan models
    print(f"\n🌊 Detected Wan series model: {model_name}")

    has_transformer_2 = hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None
    if "2.2" in model_name or has_transformer_2:
        print("🚀 Detected Wan2.2 with dual-transformer (MoE) architecture")
        print("Setting up PSA for main transformer...")
        inner_attn = set_block_sparse_attn_wan(
            pipe.transformer,
            model_name=model_name,
            inference_num=inference_num,
            attention_info=attention_info,
            verbose=verbose,
        )

        if has_transformer_2:
            print("🚀 Setting up PSA for transformer_2 - sharing processors with transformer")
            for layer_idx, block in enumerate(pipe.transformer_2.blocks):
                shared_processor = pipe.transformer.blocks[layer_idx].attn1.processor
                block.attn1.verbose = verbose
                block.attn1.inner_attention = inner_attn
                block.attn1.set_processor(shared_processor)
                if not hasattr(block.attn1, "origin_processor"):
                    block.attn1.origin_processor = pipe.transformer.blocks[layer_idx].attn1.origin_processor
    else:
        print("Setting up PSA for Wan2.1...")
        set_block_sparse_attn_wan(
            pipe.transformer,
            model_name=model_name,
            inference_num=inference_num,
            attention_info=attention_info,
            verbose=verbose,
        )

    print(f"✅ Successfully configured Pyramid Sparse Attention preset '{attention_info['name']}' for {model_name}\n")
    return attention_info


def reset_wan_attention_to_dense(
    pipe,
    include_transformer_2: bool = True,
):
    """Reset Wan pipeline attention processors to default dense implementation"""

    def _iter_blocks(target):
        if target is None or not hasattr(target, "blocks"):
            return []
        return getattr(target, "blocks", [])

    targets = []
    if hasattr(pipe, "blocks"):
        targets.append(pipe)
    else:
        if hasattr(pipe, "transformer"):
            targets.append(pipe.transformer)
        if include_transformer_2 and hasattr(pipe, "transformer_2"):
            targets.append(pipe.transformer_2)

    for module in targets:
        for block in _iter_blocks(module):
            attn = getattr(block, "attn1", None)
            if attn is None:
                continue
            if hasattr(attn, "origin_processor"):
                attn.set_processor(attn.origin_processor)
            if hasattr(attn, "inner_attention"):
                attn.inner_attention = None
            attn.verbose = False



