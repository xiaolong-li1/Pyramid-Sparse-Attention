import torch
import time
def timeit(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.time()
        ret = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()
        print(f"{func.__name__} execution took {(end - start)*1000:.4f}ms")
        return ret
    return wrapper

try:
    from matplotlib import pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    np = None

from typing import Dict, Optional, Union

def visualize_head_seq(
    data_dict: Dict[str, Union[torch.Tensor, 'np.ndarray']],
    batch_idx: int = 0,
    max_heads_per_row: int = 4,
    figsize_scale: float = 3.0,
    cmap: str = 'viridis',
    value_range: Union[str, tuple] = 'auto',
    colorbar: bool = True,
    symmetric_range: bool = True
):
    """
    Professional multi-head sequence data visualization function

    Parameters:
    - data_dict: Data dictionary {title: tensor (batch, heads, seq, seq) or (heads, seq, seq)}
    - batch_idx: Batch index (only effective when input is 4D)
    - max_heads_per_row: Maximum number of heads displayed per row
    - figsize_scale: Figure size scaling factor (base size for each subplot)
    - cmap: Color mapping scheme
    - value_range: Value range ('auto' or (vmin, vmax))
    - colorbar: Whether to display color bar
    - symmetric_range: Whether to force symmetric color range (suitable for correlation matrices)
    """

    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
    
    # Input validation and preprocessing
    processed_data = {}
    for name, data in data_dict.items():
        # Dimension processing
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        if data.ndim == 4:
            data = data[batch_idx]  # (heads, seq, seq)
        elif data.ndim == 3:
            pass  # Directly use (heads, seq, seq)
        else:
            raise ValueError(f"Input data {name} dimension error, should be 3D or 4D")

        processed_data[name] = data

    # Get key parameters
    num_heads = min(d.shape[0] for d in processed_data.values())
    num_datasets = len(processed_data)
    seq_len = next(iter(processed_data.values())).shape[-1]

    # Smart layout calculation
    rows = int(np.ceil(num_heads / max_heads_per_row))
    cols = max_heads_per_row * num_datasets

    # Create canvas
    fig, axes = plt.subplots(
        rows, 
        cols, 
        figsize=(cols * figsize_scale , 
                rows * figsize_scale),
        gridspec_kw={'wspace':0.3, 'hspace':0.1}
    )
    
    # Unified color range
    all_values = np.concatenate([d.ravel() for d in processed_data.values()])
    if value_range == 'auto':
        vmin, vmax = (all_values.min(), all_values.max())
        if symmetric_range:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound
    else:
        vmin, vmax = value_range

    # Visualization main loop
    for head_idx in range(num_heads):
        row = head_idx // max_heads_per_row
        col_start = (head_idx % max_heads_per_row) * num_datasets

        for data_idx, (name, data) in enumerate(processed_data.items()):
            ax = axes[row, col_start + data_idx] if rows > 1 else axes[col_start + data_idx]

            # Extract current head data
            current_data = data[head_idx] if data.shape[0] > 1 else data[0]

            # Draw heatmap
            im = ax.imshow(current_data, cmap=cmap, vmin=vmin, vmax=vmax)

            # Annotation settings
            ax.set_xticks([])
            ax.set_yticks([])
            if head_idx == 0:
                ax.set_title(f"{name}\nSeqLen={seq_len}", 
                           fontsize=9, pad=12, color='#2F4F4F')
                
            if data_idx == 0:
                ax.text(-0.1, 0.5, f'Head {head_idx}', 
                       rotation=90, va='center', ha='right',
                       transform=ax.transAxes, fontsize=8)
            
            # Add color bar
            if colorbar and (head_idx == num_heads-1) and (data_idx == num_datasets-1):
                cax = fig.add_axes([ax.get_position().x1+0.02,
                                  ax.get_position().y0,
                                  0.02,
                                  ax.get_position().height])
                fig.colorbar(im, cax=cax)

    # Hide empty subplots
    for r in range(rows):
        for c in range(cols):
            if (r * max_heads_per_row + c//num_datasets) >= num_heads:
                if rows > 1:
                    axes[r,c].axis('off')
                else:
                    axes[c].axis('off')

    plt.suptitle(f"Multi-Head Attention Pattern Visualization (Batch {batch_idx})", 
                y=1.02, fontsize=11, color='#2F4F4F')
    plt.tight_layout()
    plt.show()

# Usage example ---------------------------------------------------
if __name__ == "__main__":
    # Generate simulated data
    batch_size, num_heads, seq_len = 2, 6, 64

    # Raw attention matrix
    attn = torch.rand(batch_size, num_heads, seq_len, seq_len)
    # Sparse mask
    mask = (torch.rand(batch_size, 1, seq_len, seq_len) > 0.7).expand(-1, num_heads, -1, -1)
    # Correlation matrix
    corr = torch.randn(batch_size, num_heads, seq_len, seq_len)

    # Call visualization function
    visualize_head_seq(
        data_dict={
            "Raw Attention": attn
        },
        max_heads_per_row=3,
        figsize_scale=2.5,
        cmap='coolwarm',
        symmetric_range=True,
        value_range=(-3, 3)
    )

import torch
from functools import wraps

def preserve_rng_state(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save current random state
        cpu_state = torch.get_rng_state()
        cuda_states = []
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                cuda_states.append(torch.cuda.get_rng_state(device))
        try:
            # Execute decorated function
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore random state
            torch.set_rng_state(cpu_state)
            if torch.cuda.is_available():
                for device, state in enumerate(cuda_states):
                    torch.cuda.set_rng_state(state, device)
    return wrapper

import json

def analyze_and_visualize(filename='sparsity_records.json', mask_type='all_mask'):
    """Analyze and visualize sparsity data from a file for the specified mask type.
    mask_type should be one of 'inner_frame_mask', 'outer_frame_mask', or 'all_mask'.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

    # Try to load JSON file
    try:
        with open(filename, 'r') as f:
            sparsity_records = json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error parsing JSON data from file {filename}.")
        return

    # Check if file contains data of specified mask_type
    if mask_type not in sparsity_records:
        print(f"File does not contain data of type '{mask_type}'.")
        return

    records = sparsity_records[mask_type]
    if not records:
        print("No sparsity data available for analysis.")
        return

    # Extract data from records
    timesteps = [r[0] for r in records]
    layeridxs = [r[1] for r in records]
    sparsities = [r[2] for r in records]

    # Determine number of layers (assuming layer indices start from 0)
    layernum = max(layeridxs) + 1

    # Plot line chart of sparsity vs timestep for each layer
    plt.figure(figsize=(10, 6))
    for layer in range(layernum):
        layer_sparsities = [sparsities[i] for i in range(len(records)) if layeridxs[i] == layer]
        layer_timesteps = [timesteps[i] for i in range(len(records)) if layeridxs[i] == layer]
        if layer_sparsities:  # Only plot when data exists
            plt.plot(layer_timesteps, layer_sparsities, label=f'Layer {layer}')

    plt.xlabel('Timestep')
    plt.ylabel('Sparsity')
    plt.title(f'Sparsity of {mask_type} over Timesteps for Each Layer')
    plt.legend()
    plt.grid(True)
    plt.show()