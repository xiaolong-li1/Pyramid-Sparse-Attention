"""Timing utilities for measuring inference performance."""

import time
from typing import Optional
import torch


class Timer:
    """Context manager for timing code blocks with CUDA synchronization."""

    def __init__(self, name: str = "Operation", use_cuda: bool = True):
        """
        Args:
            name: Name of the operation being timed
            use_cuda: Whether to use CUDA events for timing
        """
        self.name = name
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.elapsed_ms: Optional[float] = None

        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_time = None

    def __enter__(self):
        if self.use_cuda:
            self.start_event.record()
        else:
            self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.elapsed_ms = (time.time() - self.start_time) * 1000

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed_ms is None:
            return 0.0
        return self.elapsed_ms / 1000.0

    def print_summary(self):
        """Print timing summary."""
        if self.elapsed_ms is not None:
            print(f"{self.name}: {format_time(self.elapsed_ms)}")


def format_time(ms: float) -> str:
    """Format milliseconds into human-readable string.

    Args:
        ms: Time in milliseconds

    Returns:
        Formatted string (e.g., "1.23s" or "456.78ms")
    """
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    else:
        return f"{ms:.2f}ms"
