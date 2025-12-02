"""
PSA Triton - Pyramid Block Sparse Attention

Pyramid Block Sparse Attention (PSA) is an adaptive block sparse attention mechanism
that achieves efficient attention computation through multi-level pooling strategy.
"""

from .pyramid_sparse_attention import (
    PyramidSparseAttention,
    AttentionConfig
)

__version__ = "0.1.0"

__all__ = [
    "PyramidSparseAttention",
    "AttentionConfig",
]
