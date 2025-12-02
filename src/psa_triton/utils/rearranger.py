"""
Rearranger class collection
Contains various sequence rearrangement methods: Gilbert curve, semantic-aware, STA, hybrid methods
"""

import torch
from typing import Dict, Tuple
from ..utils.gilbert3d import gilbert3d
from einops import rearrange
from ..kernels.permute_related.permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton
from ..kernels.permute_related.kmeans_utils import batch_kmeans_Euclid


class GilbertRearranger:
    """Gilbert curve-based sequence rearranger for rearranging video and text data."""
    def __init__(self, width: int, height: int, depth: int, text_length: int = 224):
        self.width = width
        self.height = height
        self.depth = depth
        self.total_elements = width * height * depth
        self.text_length = text_length

        coord_to_index = self._gilbert3d_with_index(width, height, depth)
        original_order2gilbert_order = [0] * self.total_elements
        gilbert_order2original_order = [0] * self.total_elements

        for coord_idx, org_idx in coord_to_index.items():
            original_order2gilbert_order[org_idx] = coord_idx
            gilbert_order2original_order[coord_idx] = org_idx

        # Store as CPU tensors to avoid device mismatch, will move to correct device lazily
        self.original_order2gilbert_order_cpu = torch.tensor(original_order2gilbert_order, dtype=torch.long)
        self.gilbert_order2original_order_cpu = torch.tensor(gilbert_order2original_order, dtype=torch.long)

        # Cache for device-specific tensors to avoid repeated transfers
        self._device_cache = {}

    def _get_indices_for_device(self, device: torch.device, forward: bool = True) -> torch.Tensor:
        """
        Lazy loading: get index tensor on specified device, use cache to avoid repeated transfers

        Args:
            device: Target device
            forward: True returns forward indices, False returns reverse indices
        """
        cache_key = (str(device), forward)

        if cache_key not in self._device_cache:
            if forward:
                indices = self.original_order2gilbert_order_cpu.to(device, non_blocking=True)
            else:
                indices = self.gilbert_order2original_order_cpu.to(device, non_blocking=True)
            self._device_cache[cache_key] = indices

        return self._device_cache[cache_key]

    def _gilbert3d_with_index(self, width: int, height: int, depth: int) -> Dict[int, int]:
        """Generate Gilbert curve coordinate to index mapping."""
        coord_to_index = {}
        index = 0
        def coord_to_index_func(x, y, z):
            return x + width * (y + height * z)
        for x, y, z in gilbert3d(width, height, depth):
            coord_index = coord_to_index_func(x, y, z)
            coord_to_index[coord_index] = index
            index += 1
        return coord_to_index

    def rearrange(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rearrange the video part of q, k, v tensors in Gilbert curve order."""
        seq_dim = -2

        # Automatically get the device of input tensor
        indices = self._get_indices_for_device(q.device, forward=True)

        # Handle text_length=0 case
        if self.text_length == 0:
            # No text part, directly rearrange entire sequence
            q_rearranged = q.index_select(seq_dim, indices)
            k_rearranged = k.index_select(seq_dim, indices)
            v_rearranged = v.index_select(seq_dim, indices)
            return q_rearranged, k_rearranged, v_rearranged

        # Original logic: case with text part
        text_part_q, video_part_q = q[..., :self.text_length, :], q[..., self.text_length:, :]
        text_part_k, video_part_k = k[..., :self.text_length, :], k[..., self.text_length:, :]
        text_part_v, video_part_v = v[..., :self.text_length, :], v[..., self.text_length:, :]

        q_rearranged = video_part_q.index_select(seq_dim, indices)
        k_rearranged = video_part_k.index_select(seq_dim, indices)
        v_rearranged = video_part_v.index_select(seq_dim, indices)

        return (torch.cat((q_rearranged, text_part_q), dim=seq_dim),
                torch.cat((k_rearranged, text_part_k), dim=seq_dim),
                torch.cat((v_rearranged, text_part_v), dim=seq_dim))

    def rearrange_single(self, q: torch.Tensor) -> torch.Tensor:
        """Rearrange the video part of a single tensor in Gilbert curve order."""
        seq_dim = -2

        # Automatically get the device of input tensor
        indices = self._get_indices_for_device(q.device, forward=True)

        # Handle text_length=0 case
        if self.text_length == 0:
            # No text part, directly rearrange entire sequence
            q_rearranged = q.index_select(seq_dim, indices)
            return q_rearranged

        # Original logic: case with text part
        text_part_q, video_part_q = q[..., :self.text_length, :], q[..., self.text_length:, :]

        q_rearranged = video_part_q.index_select(seq_dim, indices)

        return torch.cat((q_rearranged, text_part_q), dim=seq_dim)

    def reversed_rearrange(self, out: torch.Tensor) -> torch.Tensor:
        """Restore the video part of output tensor from Gilbert curve order to original order."""
        seq_dim = -2

        # Automatically get the device of input tensor
        indices = self._get_indices_for_device(out.device, forward=False)

        # Handle text_length=0 case
        if self.text_length == 0:
            # No text part, directly restore entire sequence
            return out.index_select(seq_dim, indices)

        # Original logic: case with text part
        video_part, text_part = out[..., :-self.text_length, :], out[..., -self.text_length:, :]
        out_reversed = video_part.index_select(seq_dim, indices)
        return torch.cat((text_part, out_reversed), dim=seq_dim)


class SemanticAwareRearranger:
    """Semantic-aware rearranger using K-means clustering for sequence rearrangement"""
    def __init__(self, num_q_centroids=200, num_k_centroids=1000,
                 kmeans_iter_init=50, kmeans_iter_step=2, layer_idx=0):
        """
        Initialize semantic-aware rearranger

        Args:
            num_q_centroids: Number of clustering centers for query
            num_k_centroids: Number of clustering centers for key
            kmeans_iter_init: Number of kmeans iterations during initialization
            kmeans_iter_step: Number of kmeans iterations per step
        """
        self.num_q_centroids = num_q_centroids
        self.num_k_centroids = num_k_centroids

        self.centroids_init = False
        self.q_centroids = None
        self.k_centroids = None

        self.kmeans_iter_init = kmeans_iter_init
        self.kmeans_iter_step = kmeans_iter_step
        self.layer_idx = layer_idx
        self.counter = 0

    def reset_clustering(self):
        if self.counter == 0:
            self.centroids_init = False
        self.counter += 1
        self.counter = self.counter % 100

    def kmeans_init(self, query, key, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_q_centroids, max_iters=self.kmeans_iter_init
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_k_centroids, max_iters=self.kmeans_iter_init
        )

        self.q_centroids = qcentroids
        self.k_centroids = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    def kmeans_step(self, query, key, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_q_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.q_centroids,
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_k_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.k_centroids,
        )

        self.q_centroids = qcentroids
        self.k_centroids = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    def kmeans_clustering(self, query, key, layer_idx):
        self.reset_clustering()
        if not self.centroids_init:
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_init(
                query, key, layer_idx
            )
            self.centroids_init = True
            print(f"Centroids initialized at layer {layer_idx}. Init step: {self.kmeans_iter_init}")
        else:
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_step(
                query, key, layer_idx
            )

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    def semantic_aware_permutation(self, query, key, value):
        cfg, num_heads, seq_len, dim = query.size()

        # 1. Kmeans clustering
        qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_clustering(
            query, key, self.layer_idx
        )

        # 2. Permute the query, key, value
        q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(query, qlabels, dim=2)
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(key, klabels, dim=2)
        v_permuted, v_sorted_indices = permute_tensor_by_labels_triton(
            value, klabels, dim=2, sorted_indices=k_sorted_indices
        )

        return q_permuted, k_permuted, v_permuted, q_sorted_indices

    def reverse_permutation(self, out, q_sorted_indices):
        # Reverse the permutation on output
        out_reversed = apply_inverse_permutation_triton(out, q_sorted_indices, dim=2)
        return out_reversed


class STARearranger:
    """Sliding Tile Attention-based sequence rearranger for tile-based rearrangement of video data.

    Text comes after video, format is [text, video]
    """
    def __init__(self, width: int, height: int, depth: int, text_length: int = 226,
                 tile_size: Tuple[int, int, int] = (13, 3, 3)):
        """
        Initialize STA rearranger

        Args:
            width: Video width
            height: Video height
            depth: Video depth (temporal dimension)
            text_length: Text sequence length
            tile_size: Tile size (T, H, W)
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.text_length = text_length
        self.tile_size = tile_size

        # Calculate number of patches
        self.patch_size = (
            depth // tile_size[0],
            height // tile_size[1],
            width // tile_size[2]
        )

        # Verify dimension matching
        assert depth % tile_size[0] == 0, f"depth {depth} must be divisible by tile_size[0] {tile_size[0]}"
        assert height % tile_size[1] == 0, f"height {height} must be divisible by tile_size[1] {tile_size[1]}"
        assert width % tile_size[2] == 0, f"width {width} must be divisible by tile_size[2] {tile_size[2]}"

        self.img_size = (depth, height, width)
        self.total_video_tokens = depth * height * width

    def rearrange(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rearrange the video part of q, k, v tensors in tile order

        Input format: [batch, heads, text_length + video_tokens, dim]
        where video_tokens = depth * height * width

        Rearrangement rule: (nt tp nh hp nw wp) -> (nt nh nw tp hp wp)
        i.e.: first by patch order (nt, nh, nw), then by intra-tile order (tp, hp, wp)
        """
        if self.text_length == 0:
            # No text, all video
            q_img = q
            k_img = k
            v_img = v
            text_part_q = None
            text_part_k = None
            text_part_v = None
        else:
            # text first, video second: [batch, heads, text+video, dim]
            text_part_q = q[:, :, :self.text_length, :]
            text_part_k = k[:, :, :self.text_length, :]
            text_part_v = v[:, :, :self.text_length, :]

            q_img = q[:, :, self.text_length:, :]
            k_img = k[:, :, self.text_length:, :]
            v_img = v[:, :, self.text_length:, :]

        # Rearrange video part: (nt tp nh hp nw wp) -> (nt nh nw tp hp wp)
        tp, hp, wp = self.tile_size
        nt, nh, nw = self.patch_size

        q_img_rearranged = rearrange(
            q_img,
            "b H (nt tp nh hp nw wp) d -> b H (nt nh nw tp hp wp) d",
            tp=tp, hp=hp, wp=wp, nt=nt, nh=nh, nw=nw
        )
        k_img_rearranged = rearrange(
            k_img,
            "b H (nt tp nh hp nw wp) d -> b H (nt nh nw tp hp wp) d",
            tp=tp, hp=hp, wp=wp, nt=nt, nh=nh, nw=nw
        )
        v_img_rearranged = rearrange(
            v_img,
            "b H (nt tp nh hp nw wp) d -> b H (nt nh nw tp hp wp) d",
            tp=tp, hp=hp, wp=wp, nt=nt, nh=nh, nw=nw
        )

        # Concatenate text and video back
        if self.text_length == 0:
            return q_img_rearranged, k_img_rearranged, v_img_rearranged
        else:
            q_result = torch.cat([text_part_q, q_img_rearranged], dim=2)
            k_result = torch.cat([text_part_k, k_img_rearranged], dim=2)
            v_result = torch.cat([text_part_v, v_img_rearranged], dim=2)

            return (q_result.contiguous(),
                    k_result.contiguous(),
                    v_result.contiguous())

    def reversed_rearrange(self, out: torch.Tensor) -> torch.Tensor:
        """
        Restore the video part of output tensor from tile order to original order

        Reverse rearrangement: (nt nh nw tp hp wp) -> (nt tp nh hp nw wp)
        """
        if self.text_length == 0:
            # No text, all video
            out_img = out
            text_part = None
        else:
            # text first, video second
            text_part = out[:, :, :self.text_length, :]
            out_img = out[:, :, self.text_length:, :]

        # Reverse rearrange video part: (nt nh nw tp hp wp) -> (nt tp nh hp nw wp)
        tp, hp, wp = self.tile_size
        nt, nh, nw = self.patch_size

        out_img_reversed = rearrange(
            out_img,
            "b H (nt nh nw tp hp wp) d -> b H (nt tp nh hp nw wp) d",
            tp=tp, hp=hp, wp=wp, nt=nt, nh=nh, nw=nw
        )

        # Concatenate text and video back
        if self.text_length == 0:
            return out_img_reversed
        else:
            return torch.cat([text_part, out_img_reversed], dim=2)


class HybridRearranger:
    """Hybrid rearranger: uses Gilbert curve rearrangement for q, semantic-aware rearrangement for k.

    - q: Gilbert spatial curve rearrangement (text and video processed separately)
    - k, v: Semantic clustering rearrangement (entire sequence processed together)
    - out: Restored using Gilbert's reverse logic (because output follows q's order)
    """
    def __init__(self, width: int, height: int, depth: int, text_length: int = 226,
                 num_k_centroids: int = 1000, kmeans_iter_init: int = 50,
                 kmeans_iter_step: int = 2, layer_idx: int = 0):
        """
        Initialize hybrid rearranger

        Args:
            width: Video width
            height: Video height
            depth: Video depth (temporal dimension)
            text_length: Text sequence length
            num_k_centroids: Number of clustering centers for k
            kmeans_iter_init: Number of kmeans iterations during initialization
            kmeans_iter_step: Number of kmeans iterations per step
            layer_idx: Layer index
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.text_length = text_length
        self.total_elements = width * height * depth
        self.layer_idx = layer_idx

        # Initialize Gilbert mapping (for q)
        coord_to_index = self._gilbert3d_with_index(width, height, depth)
        original_order2gilbert_order = [0] * self.total_elements
        gilbert_order2original_order = [0] * self.total_elements

        for coord_idx, org_idx in coord_to_index.items():
            original_order2gilbert_order[org_idx] = coord_idx
            gilbert_order2original_order[coord_idx] = org_idx

        # Store as CPU tensors to avoid device mismatch, will move to correct device lazily
        self.original_order2gilbert_order_cpu = torch.tensor(original_order2gilbert_order, dtype=torch.long)
        self.gilbert_order2original_order_cpu = torch.tensor(gilbert_order2original_order, dtype=torch.long)

        # Cache for device-specific tensors to avoid repeated transfers
        self._device_cache = {}

        # Initialize SemanticAware parameters (for k)
        self.num_k_centroids = num_k_centroids
        self.centroids_init = False
        self.k_centroids = None
        self.kmeans_iter_init = kmeans_iter_init
        self.kmeans_iter_step = kmeans_iter_step
        self.counter = 0

    def _gilbert3d_with_index(self, width: int, height: int, depth: int) -> Dict[int, int]:
        """Generate Gilbert curve coordinate to index mapping."""
        coord_to_index = {}
        index = 0
        def coord_to_index_func(x, y, z):
            return x + width * (y + height * z)
        for x, y, z in gilbert3d(width, height, depth):
            coord_index = coord_to_index_func(x, y, z)
            coord_to_index[coord_index] = index
            index += 1
        return coord_to_index

    def _get_indices_for_device(self, device: torch.device, forward: bool = True) -> torch.Tensor:
        """
        Lazy loading: get index tensor on specified device, use cache to avoid repeated transfers

        Args:
            device: Target device
            forward: True returns forward indices, False returns reverse indices
        """
        cache_key = (str(device), forward)

        if cache_key not in self._device_cache:
            if forward:
                indices = self.original_order2gilbert_order_cpu.to(device, non_blocking=True)
            else:
                indices = self.gilbert_order2original_order_cpu.to(device, non_blocking=True)
            self._device_cache[cache_key] = indices

        return self._device_cache[cache_key]

    def reset_clustering(self):
        """Reset clustering counter"""
        if self.counter == 0:
            self.centroids_init = False
        self.counter += 1
        self.counter = self.counter % 100

    def rearrange_q_gilbert(self, q: torch.Tensor) -> torch.Tensor:
        """Rearrange q using Gilbert curve (text and video processed separately)"""
        seq_dim = -2

        # Automatically get the device of input tensor
        indices = self._get_indices_for_device(q.device, forward=True)

        # Handle text_length=0 case
        if self.text_length == 0:
            # No text part, directly rearrange entire sequence
            return q.index_select(seq_dim, indices)

        # Has text part: separate text and video
        text_part_q = q[..., :self.text_length, :]
        video_part_q = q[..., self.text_length:, :]

        # Rearrange video part with Gilbert
        q_video_rearranged = video_part_q.index_select(seq_dim, indices)

        # Concatenate: video first, text second
        return torch.cat((q_video_rearranged, text_part_q), dim=seq_dim)

    def rearrange_k_semantic(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rearrange k and v using semantic clustering (entire sequence processed together)"""
        cfg, num_heads, seq_len, dim = k.size()

        # 1. Reset clustering count
        self.reset_clustering()

        # 2. Kmeans clustering
        if not self.centroids_init:
            # Initialize clustering
            klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
                k.view(cfg * num_heads, seq_len, dim),
                n_clusters=self.num_k_centroids,
                max_iters=self.kmeans_iter_init
            )
            self.k_centroids = kcentroids
            self.centroids_init = True
            print(f"K Centroids initialized at layer {self.layer_idx}. Init step: {self.kmeans_iter_init}")
        else:
            # Use existing clustering centers
            klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
                k.view(cfg * num_heads, seq_len, dim),
                n_clusters=self.num_k_centroids,
                max_iters=self.kmeans_iter_step,
                init_centroids=self.k_centroids
            )
            self.k_centroids = kcentroids

        # 3. Rearrange k and v by label
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(k, klabels, dim=2)
        v_permuted, _ = permute_tensor_by_labels_triton(v, klabels, dim=2, sorted_indices=k_sorted_indices)

        return k_permuted, v_permuted, k_sorted_indices

    def rearrange(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hybrid rearrangement

        Args:
            q: query tensor [batch, heads, seq, dim]
            k: key tensor [batch, heads, seq, dim]
            v: value tensor [batch, heads, seq, dim]

        Returns:
            q_rearranged: q after Gilbert rearrangement
            k_rearranged: k after semantic clustering rearrangement
            v_rearranged: v after semantic clustering rearrangement
        """
        # Use Gilbert rearrangement for q
        q_rearranged = self.rearrange_q_gilbert(q)

        # Use semantic clustering rearrangement for k, v
        k_rearranged, v_rearranged, _ = self.rearrange_k_semantic(k, v)

        return q_rearranged, k_rearranged, v_rearranged

    def reversed_rearrange(self, out: torch.Tensor) -> torch.Tensor:
        """
        Restore output from Gilbert order to original order

        Args:
            out: Attention output tensor [batch, heads, seq, dim]

        Returns:
            Output restored to original order
        """
        seq_dim = -2

        # Automatically get the device of input tensor
        indices = self._get_indices_for_device(out.device, forward=False)

        # Handle text_length=0 case
        if self.text_length == 0:
            # No text part, directly restore entire sequence
            return out.index_select(seq_dim, indices)

        # Has text part: separate video and text
        video_part = out[..., :-self.text_length, :]
        text_part = out[..., -self.text_length:, :]

        # Restore video part using Gilbert reverse
        out_reversed = video_part.index_select(seq_dim, indices)

        # Concatenate: text first, video second (restore original order)
        return torch.cat((text_part, out_reversed), dim=seq_dim)
