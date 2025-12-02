"""
PSA Logging System - Records configuration and sparsity metrics for Pyramid Sparse Attention
"""
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import numpy as np


class PSALogger:
    """
    Comprehensive logger for Pyramid Adaptive Block Sparse Attention.
    Tracks configuration, per-layer sparsity, sim_mask statistics, and provides summary statistics.
    """

    def __init__(
        self,
        log_dir: str,
        config: Any,  # AttentionConfig
        model_type: str = "wan",
        layer_num: int = 42,
        inference_num: int = 50,
        session_name: Optional[str] = None,
    ):
        """
        Initialize PSA Logger.

        Args:
            log_dir: Directory to save logs
            config: AttentionConfig instance
            model_type: Model type ('wan', 'cogvideo', etc.)
            layer_num: Number of layers in the model
            inference_num: Number of inference steps
            session_name: Optional session identifier
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.model_type = model_type
        self.layer_num = layer_num
        self.inference_num = inference_num

        # Create session timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"psa_session_{timestamp}"

        # Create session directory
        self.session_dir = self.log_dir / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Initialize log files
        self.config_file = self.session_dir / "config.json"
        self.sparsity_file = self.session_dir / "sparsity.jsonl"
        self.summary_file = self.session_dir / "summary.txt"

        # Open sparsity log file handle
        self.sparsity_handle = open(self.sparsity_file, "a", encoding="utf-8")

        # Tracking variables
        self.layer_stats: Dict[int, Dict[str, Any]] = {}
        self.global_counter = 0

        # Write initial configuration
        self._write_config()
        self._write_startup_banner()

    def _write_config(self):
        """Write configuration to JSON file."""
        config_dict = {
            "session_name": self.session_name,
            "timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "layer_num": self.layer_num,
            "inference_num": self.inference_num,
            "config": asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else str(self.config),
        }

        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        print(f"[PSA Logger] Configuration saved to: {self.config_file}")

    def _write_startup_banner(self):
        """Write a startup banner with config details."""
        banner_lines = [
            "=" * 80,
            f"PSA Logger - Session: {self.session_name}",
            "=" * 80,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Log Directory: {self.log_dir}",
            f"Model Type: {self.model_type}",
            f"Layer Num: {self.layer_num}",
            f"Inference Num: {self.inference_num}",
            "",
            "Configuration:",
            f"  - mask_mode: {self.config.mask_mode}",
            f"  - mask_ratios: {self.config.mask_ratios}",
            f"  - attn_impl: {self.config.attn_impl}",
            f"  - warmup_steps: {self.config.warmup_steps}",
            "",
            "Video Parameters:",
            f"  - width: {self.config.width}",
            f"  - height: {self.config.height}",
            f"  - depth: {self.config.depth}",
            f"  - text_length: {self.config.text_length}",
            "",
            "Block Parameters:",
            f"  - block_m: {self.config.block_m}",
            f"  - block_n: {self.config.block_n}",
            f"  - tile_n: {self.config.tile_n}",
            f"  - use_sim_mask: {getattr(self.config, 'use_sim_mask', True)}",
            "",
            "Sim Mask Thresholds (Adaptive Pooling):",
            f"  - sim_2x_threshold: {self.config.sim_2x_threshold}",
            f"  - sim_4x_threshold: {self.config.sim_4x_threshold}",
            f"  - sim_8x_threshold: {self.config.sim_8x_threshold}",
        ]

        if self.config.use_rearrange:
            banner_lines.extend([
                "",
                "Rearrange Configuration:",
                f"  - rearrange_method: {self.config.rearrange_method}",
            ])
            if self.config.rearrange_method == "STA" and self.config.tile_size:
                banner_lines.append(f"  - tile_size: {self.config.tile_size}")

        banner_lines.extend([
            "=" * 80,
            ""
        ])

        banner = "\n".join(banner_lines)
        print(banner)

        # Also write to summary file
        with open(self.summary_file, "w", encoding="utf-8") as f:
            f.write(banner + "\n")

    def _compute_sim_mask_stats(self, sim_mask: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics for sim_mask showing distribution of pooling granularities.

        Args:
            sim_mask: Tensor with values in {1, 2, 4, 8} indicating pooling granularity

        Returns:
            Dictionary with percentage of each pooling level
        """
        if sim_mask is None:
            return {
                "pool_1x_ratio": 0.0,
                "pool_2x_ratio": 0.0,
                "pool_4x_ratio": 0.0,
                "pool_8x_ratio": 0.0,
            }

        # Convert to numpy for easier computation
        sim_mask_np = sim_mask.cpu().numpy() if isinstance(sim_mask, torch.Tensor) else sim_mask
        total_elements = sim_mask_np.size

        # Count each pooling level
        pool_1x_count = np.sum(sim_mask_np == 1)
        pool_2x_count = np.sum(sim_mask_np == 2)
        pool_4x_count = np.sum(sim_mask_np == 4)
        pool_8x_count = np.sum(sim_mask_np == 8)

        return {
            "pool_1x_ratio": float(pool_1x_count / total_elements) if total_elements > 0 else 0.0,
            "pool_2x_ratio": float(pool_2x_count / total_elements) if total_elements > 0 else 0.0,
            "pool_4x_ratio": float(pool_4x_count / total_elements) if total_elements > 0 else 0.0,
            "pool_8x_ratio": float(pool_8x_count / total_elements) if total_elements > 0 else 0.0,
        }

    def log_sparsity(
        self,
        layer_idx: int,
        sparsity: float,
        per_head_density: List[float],
        sequence_length: int,
        batch_size: int = 1,
        num_heads: int = -1,
        sim_mask: Optional[torch.Tensor] = None,
        is_warmup: bool = False,
    ):
        """
        Log sparsity information for a single forward pass.

        Args:
            layer_idx: Current layer index
            sparsity: Overall sparsity value (0-1, higher = more sparse)
            per_head_density: Density per attention head
            sequence_length: Input sequence length
            batch_size: Batch size
            num_heads: Number of attention heads
            sim_mask: Similarity mask tensor indicating pooling granularity
            is_warmup: Whether this forward pass is in warmup phase
        """
        self.global_counter += 1

        # Initialize layer stats if needed
        if layer_idx not in self.layer_stats:
            self.layer_stats[layer_idx] = {
                "count": 0,
                "total_sparsity": 0.0,
                "min_sparsity": float("inf"),
                "max_sparsity": 0.0,
                "sparsity_history": [],
                # Statistics excluding warmup (sparsity > 0, i.e., density < 1)
                "non_warmup_count": 0,
                "non_warmup_total_sparsity": 0.0,
                "non_warmup_min_sparsity": float("inf"),
                "non_warmup_max_sparsity": 0.0,
                "non_warmup_sparsity_history": [],
                "sim_mask_stats": {
                    "pool_1x_sum": 0.0,
                    "pool_2x_sum": 0.0,
                    "pool_4x_sum": 0.0,
                    "pool_8x_sum": 0.0,
                    "count": 0,  # Separately count sim_mask occurrences (excluding warmup)
                }
            }

        # Update layer statistics
        stats = self.layer_stats[layer_idx]
        stats["count"] += 1
        stats["total_sparsity"] += sparsity
        stats["min_sparsity"] = min(stats["min_sparsity"], sparsity)
        stats["max_sparsity"] = max(stats["max_sparsity"], sparsity)
        stats["sparsity_history"].append(sparsity)

        # Keep only last 100 entries in history
        if len(stats["sparsity_history"]) > 100:
            stats["sparsity_history"] = stats["sparsity_history"][-100:]

        # Update non-warmup statistics (exclude entries with sparsity == 0, i.e., density == 1)
        if sparsity > 0:
            stats["non_warmup_count"] += 1
            stats["non_warmup_total_sparsity"] += sparsity
            stats["non_warmup_min_sparsity"] = min(stats["non_warmup_min_sparsity"], sparsity)
            stats["non_warmup_max_sparsity"] = max(stats["non_warmup_max_sparsity"], sparsity)
            stats["non_warmup_sparsity_history"].append(sparsity)

            # Keep only last 100 entries in non-warmup history
            if len(stats["non_warmup_sparsity_history"]) > 100:
                stats["non_warmup_sparsity_history"] = stats["non_warmup_sparsity_history"][-100:]

        # Compute sim_mask statistics (only for non-warmup)
        sim_mask_stats = self._compute_sim_mask_stats(sim_mask)
        sim_mask_enabled = getattr(self.config, "use_sim_mask", True)

        # Update cumulative sim_mask stats (only if not warmup)
        if sim_mask_enabled and not is_warmup and sim_mask is not None:
            stats["sim_mask_stats"]["pool_1x_sum"] += sim_mask_stats["pool_1x_ratio"]
            stats["sim_mask_stats"]["pool_2x_sum"] += sim_mask_stats["pool_2x_ratio"]
            stats["sim_mask_stats"]["pool_4x_sum"] += sim_mask_stats["pool_4x_ratio"]
            stats["sim_mask_stats"]["pool_8x_sum"] += sim_mask_stats["pool_8x_ratio"]
            stats["sim_mask_stats"]["count"] += 1

        # Write to JSONL file
        entry = {
            "global_step": self.global_counter,
            "timestamp": datetime.now().isoformat(),
            "layer_idx": layer_idx,
            "sparsity": round(sparsity, 6),
            "density": round(1 - sparsity, 6),
            "per_head_density": [round(d, 6) for d in per_head_density],
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "num_heads": num_heads if num_heads > 0 else len(per_head_density),
            "is_warmup": is_warmup,
            "use_sim_mask": sim_mask_enabled,
            "sim_mask_stats": {
                "pool_1x_ratio": round(sim_mask_stats["pool_1x_ratio"], 6),
                "pool_2x_ratio": round(sim_mask_stats["pool_2x_ratio"], 6),
                "pool_4x_ratio": round(sim_mask_stats["pool_4x_ratio"], 6),
                "pool_8x_ratio": round(sim_mask_stats["pool_8x_ratio"], 6),
            }
        }

        self.sparsity_handle.write(json.dumps(entry) + "\n")
        self.sparsity_handle.flush()

    def print_progress(self, layer_idx: int, interval: int = 200):
        """
        Print progress statistics at regular intervals.

        Args:
            layer_idx: Current layer index
            interval: Print every N steps
        """
        if layer_idx not in self.layer_stats:
            return

        stats = self.layer_stats[layer_idx]
        if stats["count"] % interval != 0:
            return

        avg_sparsity = stats["total_sparsity"] / stats["count"]
        current_sparsity = stats["sparsity_history"][-1] if stats["sparsity_history"] else 0.0

        print(
            f"[Layer {layer_idx:2d}] "
            f"Step {stats['count']:4d} | "
            f"Avg Sparsity: {avg_sparsity:.4f} | "
            f"Current: {current_sparsity:.4f} | "
            f"Range: [{stats['min_sparsity']:.4f}, {stats['max_sparsity']:.4f}]"
        )

    def write_summary(self):
        """Write comprehensive summary statistics."""
        if not self.layer_stats:
            return

        summary_lines = [
            "",
            "=" * 80,
            f"PSA Sparsity Summary - {self.session_name}",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Forward Passes: {self.global_counter}",
            "",
            "Per-Layer Statistics:",
            "-" * 80,
        ]

        # Table header
        summary_lines.append(
            f"{'Layer':<8} {'Count':<8} {'Avg Sparsity':<15} {'Min':<10} {'Max':<10} {'StdDev':<10}"
        )
        summary_lines.append("-" * 80)

        # Per-layer statistics
        for layer_idx in sorted(self.layer_stats.keys()):
            stats = self.layer_stats[layer_idx]
            avg_sparsity = stats["total_sparsity"] / stats["count"]

            # Calculate standard deviation
            history = stats["sparsity_history"]
            if len(history) > 1:
                mean = sum(history) / len(history)
                variance = sum((x - mean) ** 2 for x in history) / len(history)
                std_dev = variance ** 0.5
            else:
                std_dev = 0.0

            summary_lines.append(
                f"{layer_idx:<8} {stats['count']:<8} "
                f"{avg_sparsity:<15.6f} "
                f"{stats['min_sparsity']:<10.6f} "
                f"{stats['max_sparsity']:<10.6f} "
                f"{std_dev:<10.6f}"
            )

        summary_lines.extend([
            "-" * 80,
            "",
            "Overall Statistics:",
            "-" * 80,
        ])

        # Overall statistics
        all_sparsities = []
        total_count = 0
        total_sparsity = 0.0

        for stats in self.layer_stats.values():
            all_sparsities.extend(stats["sparsity_history"])
            total_count += stats["count"]
            total_sparsity += stats["total_sparsity"]

        if all_sparsities:
            overall_avg = total_sparsity / total_count
            overall_min = min(all_sparsities)
            overall_max = max(all_sparsities)

            mean = sum(all_sparsities) / len(all_sparsities)
            variance = sum((x - mean) ** 2 for x in all_sparsities) / len(all_sparsities)
            overall_std = variance ** 0.5

            summary_lines.extend([
                f"Average Sparsity (all layers): {overall_avg:.6f}",
                f"Average Density (all layers): {1 - overall_avg:.6f}",
                f"Sparsity Range: [{overall_min:.6f}, {overall_max:.6f}]",
                f"Standard Deviation: {overall_std:.6f}",
                f"Total Samples: {len(all_sparsities)}",
            ])

        # Non-warmup statistics (excluding sparsity == 0, i.e., density == 1)
        summary_lines.extend([
            "",
            "Non-Warmup Statistics (excluding density=1.0 steps):",
            "-" * 80,
        ])

        non_warmup_all_sparsities = []
        non_warmup_total_count = 0
        non_warmup_total_sparsity = 0.0

        for stats in self.layer_stats.values():
            non_warmup_all_sparsities.extend(stats["non_warmup_sparsity_history"])
            non_warmup_total_count += stats["non_warmup_count"]
            non_warmup_total_sparsity += stats["non_warmup_total_sparsity"]

        if non_warmup_all_sparsities:
            non_warmup_avg = non_warmup_total_sparsity / non_warmup_total_count
            non_warmup_min = min(non_warmup_all_sparsities)
            non_warmup_max = max(non_warmup_all_sparsities)

            non_warmup_mean = sum(non_warmup_all_sparsities) / len(non_warmup_all_sparsities)
            non_warmup_variance = sum((x - non_warmup_mean) ** 2 for x in non_warmup_all_sparsities) / len(non_warmup_all_sparsities)
            non_warmup_std = non_warmup_variance ** 0.5

            # Calculate warmup steps count
            warmup_count = total_count - non_warmup_total_count

            summary_lines.extend([
                f"Average Sparsity (non-warmup): {non_warmup_avg:.6f}",
                f"Average Density (non-warmup): {1 - non_warmup_avg:.6f}",
                f"Sparsity Range: [{non_warmup_min:.6f}, {non_warmup_max:.6f}]",
                f"Standard Deviation: {non_warmup_std:.6f}",
                f"Non-Warmup Samples: {len(non_warmup_all_sparsities)}",
                f"Warmup Samples (density=1.0): {warmup_count}",
            ])
        else:
            summary_lines.append("No non-warmup samples recorded (all steps were warmup).")

        # Add sim_mask statistics summary
        sim_mask_enabled = getattr(self.config, "use_sim_mask", True)
        if sim_mask_enabled:
            summary_lines.extend([
                "",
                "Sim Mask Statistics (Average Pooling Distribution):",
                "-" * 80,
            ])

            # Table header for sim_mask stats
            summary_lines.append(
                f"{'Layer':<8} {'1x Pool %':<12} {'2x Pool %':<12} {'4x Pool %':<12} {'8x Pool %':<12}"
            )
            summary_lines.append("-" * 80)

            # Per-layer sim_mask statistics
            total_pool_1x = 0.0
            total_pool_2x = 0.0
            total_pool_4x = 0.0
            total_pool_8x = 0.0
            total_sim_mask_count = 0

            for layer_idx in sorted(self.layer_stats.keys()):
                stats = self.layer_stats[layer_idx]
                sim_mask_count = stats["sim_mask_stats"]["count"]

                if sim_mask_count > 0:
                    avg_1x = stats["sim_mask_stats"]["pool_1x_sum"] / sim_mask_count * 100
                    avg_2x = stats["sim_mask_stats"]["pool_2x_sum"] / sim_mask_count * 100
                    avg_4x = stats["sim_mask_stats"]["pool_4x_sum"] / sim_mask_count * 100
                    avg_8x = stats["sim_mask_stats"]["pool_8x_sum"] / sim_mask_count * 100

                    summary_lines.append(
                        f"{layer_idx:<8} {avg_1x:<12.2f} {avg_2x:<12.2f} {avg_4x:<12.2f} {avg_8x:<12.2f}"
                    )

                    # Accumulate for overall statistics
                    total_pool_1x += stats["sim_mask_stats"]["pool_1x_sum"]
                    total_pool_2x += stats["sim_mask_stats"]["pool_2x_sum"]
                    total_pool_4x += stats["sim_mask_stats"]["pool_4x_sum"]
                    total_pool_8x += stats["sim_mask_stats"]["pool_8x_sum"]
                    total_sim_mask_count += sim_mask_count

            # Add overall sim_mask statistics
            summary_lines.append("-" * 80)
            if total_sim_mask_count > 0:
                overall_pool_1x = total_pool_1x / total_sim_mask_count * 100
                overall_pool_2x = total_pool_2x / total_sim_mask_count * 100
                overall_pool_4x = total_pool_4x / total_sim_mask_count * 100
                overall_pool_8x = total_pool_8x / total_sim_mask_count * 100

                summary_lines.append(
                    f"{'Overall':<8} {overall_pool_1x:<12.2f} {overall_pool_2x:<12.2f} {overall_pool_4x:<12.2f} {overall_pool_8x:<12.2f}"
                )
                summary_lines.append("")
                summary_lines.append(f"Overall Pooling Distribution (all layers average):")
                summary_lines.append(f"  1x Pooling (no pooling):  {overall_pool_1x:.2f}%")
                summary_lines.append(f"  2x Pooling:               {overall_pool_2x:.2f}%")
                summary_lines.append(f"  4x Pooling:               {overall_pool_4x:.2f}%")
                summary_lines.append(f"  8x Pooling:               {overall_pool_8x:.2f}%")
                summary_lines.append(f"  Total:                    {overall_pool_1x + overall_pool_2x + overall_pool_4x + overall_pool_8x:.2f}%")
                summary_lines.append(f"  Total Sim Mask Samples:   {total_sim_mask_count}")
        else:
            summary_lines.extend([
                "",
                "Sim Mask Statistics: disabled (use_sim_mask = False)",
                "-" * 80,
            ])

        summary_lines.extend([
            "=" * 80,
            ""
        ])

        summary = "\n".join(summary_lines)

        # Print to console
        print(summary)

        # Append to summary file
        with open(self.summary_file, "a", encoding="utf-8") as f:
            f.write(summary)

        print(f"[PSA Logger] Summary saved to: {self.summary_file}")

    def close(self):
        """Close log files and write final summary."""
        if self.sparsity_handle is not None:
            self.sparsity_handle.close()
            self.sparsity_handle = None

        # Write final summary
        self.write_summary()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "sparsity_handle") and self.sparsity_handle is not None:
            self.sparsity_handle.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
