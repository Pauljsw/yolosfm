"""Label fusion utilities that fuse per-pixel logits into a 3D voxel grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from .voxel_grid import VoxelGrid


@dataclass
class CameraModel:
    """Lightweight container for intrinsics and pose."""

    intrinsics: np.ndarray
    pose: np.ndarray

    def __post_init__(self) -> None:
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        if self.intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must be a 3x3 matrix")

        self.pose = np.asarray(self.pose, dtype=np.float64)
        if self.pose.shape == (3, 4):
            # Convert to homogeneous matrix
            last_row = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
            self.pose = np.vstack([self.pose, last_row])
        if self.pose.shape != (4, 4):
            raise ValueError("pose must be a 4x4 homogeneous transform or 3x4 [R|t]")

    @property
    def rotation(self) -> np.ndarray:
        return self.pose[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.pose[:3, 3]


class LabelFusion:
    """Fuse 2D semantic predictions into a 3D voxel grid."""

    def __init__(
        self,
        voxel_size: float,
        num_classes: int,
        *,
        origin: Optional[np.ndarray] = None,
        bounds: Optional[np.ndarray] = None,
        depth_epsilon: float = 1e-3,
    ) -> None:
        self.grid = VoxelGrid(
            voxel_size,
            num_classes=num_classes,
            origin=None if origin is None else np.asarray(origin, dtype=np.float64),
            bounds=None if bounds is None else np.asarray(bounds, dtype=np.float64),
        )
        self.num_classes = num_classes
        self.depth_epsilon = float(depth_epsilon)
        self.stats: Dict[str, int] = {
            'num_views': 0,
            'num_voxels': 0,
            'num_pixels': 0,
        }

    def _backproject(
        self,
        u: np.ndarray,
        v: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return np.stack([x, y, z], axis=-1)

    def fuse_mask(
        self,
        mask_logits: np.ndarray,
        depth_map: np.ndarray,
        camera: CameraModel,
        *,
        mask: Optional[np.ndarray] = None,
        pixel_weights: Optional[np.ndarray] = None,
    ) -> int:
        """Fuse a single mask's logits into the voxel grid.

        Args:
            mask_logits: Array of shape (H, W, C) containing per-class logits.
            depth_map: Aligned depth map of shape (H, W) in metres.
            camera: :class:`CameraModel` describing intrinsics and pose.
            mask: Optional binary mask selecting pixels to fuse. If ``None`` all
                pixels with valid depth are considered.
            pixel_weights: Optional per-pixel weights of shape (H, W).

        Returns:
            Number of voxels updated by this call.
        """

        mask_logits = np.asarray(mask_logits, dtype=np.float64)
        if mask_logits.ndim != 3 or mask_logits.shape[2] != self.num_classes:
            raise ValueError(
                "mask_logits must have shape (H, W, num_classes)"
            )

        depth_map = np.asarray(depth_map, dtype=np.float64)
        if depth_map.shape != mask_logits.shape[:2]:
            raise ValueError("depth_map must match mask spatial dimensions")

        if mask is None:
            valid = depth_map > 0
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != depth_map.shape:
                raise ValueError("mask must match depth_map shape")
            valid = mask & (depth_map > 0)

        if not np.any(valid):
            return 0

        v_coords, u_coords = np.nonzero(valid)
        depths = depth_map[v_coords, u_coords]
        logits = mask_logits[v_coords, u_coords]

        if pixel_weights is not None:
            pixel_weights = np.asarray(pixel_weights, dtype=np.float64)
            if pixel_weights.shape != depth_map.shape:
                raise ValueError("pixel_weights must match depth_map shape")
            weights = pixel_weights[v_coords, u_coords]
        else:
            weights = np.ones_like(depths)

        points_cam = self._backproject(u_coords.astype(np.float64), v_coords.astype(np.float64), depths, camera.intrinsics)
        points_world = (camera.rotation @ points_cam.T).T + camera.translation

        voxel_indices = self.grid.world_to_voxel(points_world)

        # Visibility check: retain closest depth for each voxel
        voxel_depths: Dict[tuple, float] = {}
        voxel_logits: Dict[tuple, np.ndarray] = {}
        voxel_weights: Dict[tuple, float] = {}

        for idx, depth, logit_vec, weight in zip(voxel_indices, depths, logits, weights):
            key = tuple(idx.tolist())
            best_depth = voxel_depths.get(key)
            if best_depth is None or depth + self.depth_epsilon < best_depth:
                voxel_depths[key] = float(depth)
                voxel_logits[key] = logit_vec.copy()
                voxel_weights[key] = float(weight)
            elif abs(depth - best_depth) <= self.depth_epsilon:
                voxel_logits[key] += logit_vec
                voxel_weights[key] += float(weight)

        if not voxel_logits:
            return 0

        indices = np.array(list(voxel_logits.keys()), dtype=np.int64)
        fused_logits = np.stack(list(voxel_logits.values()), axis=0)
        fused_weights = [voxel_weights[key] for key in voxel_logits.keys()]

        self.grid.bulk_accumulate(indices, fused_logits, fused_weights)

        self.stats['num_views'] += 1
        self.stats['num_voxels'] += len(indices)
        self.stats['num_pixels'] += len(v_coords)

        return len(indices)

    def fuse_masks(
        self,
        masks: Iterable[Dict[str, np.ndarray]],
        depth_map: np.ndarray,
        camera: CameraModel,
    ) -> None:
        """Fuse a collection of masks that share the same depth and camera."""

        for mask_dict in masks:
            mask = mask_dict.get('mask')
            logits = mask_dict['logits']
            weights = mask_dict.get('weights')
            self.fuse_mask(logits, depth_map, camera, mask=mask, pixel_weights=weights)

    def export_probabilities(self) -> Dict[str, np.ndarray]:
        """Return voxel centres and per-class probabilities."""

        indices, logits = self.grid.get_logits_array()
        if indices.size == 0:
            return {
                'voxel_indices': indices,
                'voxel_centres': np.zeros((0, 3), dtype=np.float64),
                'logits': logits,
                'probabilities': np.zeros((0, self.num_classes), dtype=np.float64),
            }

        centres = self.grid.voxel_to_world(indices)
        probs = self._softmax(logits)
        return {
            'voxel_indices': indices,
            'voxel_centres': centres,
            'logits': logits,
            'probabilities': probs,
        }

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("values must be a 2D array")
        shifted = values - values.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
