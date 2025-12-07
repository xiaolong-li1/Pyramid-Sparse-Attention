"""
PSA Triton - Pyramid Sparse Attention

Pyramid Sparse Attention (PSA) is an adaptive block sparse attention mechanism
that achieves efficient attention computation through multi-level pooling strategy.
"""

# Plug-and-play attention (recommended for simple usage)
from .attention import (
    PSAAttention,
    PSAConfig,
    psa_attention,
)

# Full-featured attention (for video generation with warmup, rearrangement, etc.)
from .pyramid_sparse_attention import (
    PyramidSparseAttention,
    AttentionConfig,
)

__version__ = "0.1.0"

__all__ = [
    # Plug-and-play (simple)
    "PSAAttention",
    "PSAConfig", 
    "psa_attention",
    # Full-featured (video generation)
    "PyramidSparseAttention",
    "AttentionConfig",
]
