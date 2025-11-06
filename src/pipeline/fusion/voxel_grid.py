"""Utility classes for managing a sparse probabilistic voxel grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np


@dataclass
class VoxelStats:
    """Statistics stored for each voxel during fusion."""

    num_classes: int
    logits: np.ndarray = field(init=False)
    weight: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:  # pragma: no cover - simple allocation
        self.logits = np.zeros(self.num_classes, dtype=np.float64)

    def accumulate(self, logits: np.ndarray, weight: float = 1.0) -> None:
        """Accumulate class logits for the voxel."""

        if logits.shape[-1] != self.logits.shape[0]:  # pragma: no cover - defensive
            raise ValueError(
                f"Expected {self.logits.shape[0]} logits, received {logits.shape[-1]}"
            )

        self.logits += logits * weight
        self.weight += float(weight)
        self.count += 1


class VoxelGrid:
    """Sparse voxel grid that stores accumulated class logits."""

    def __init__(
        self,
        voxel_size: float,
        *,
        num_classes: int,
        origin: np.ndarray | None = None,
        bounds: np.ndarray | None = None,
    ) -> None:
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")

        self.voxel_size = float(voxel_size)
        self.num_classes = int(num_classes)
        self._origin = None if origin is None else np.asarray(origin, dtype=np.float64)
        self._bounds = None if bounds is None else np.asarray(bounds, dtype=np.float64)
        self._voxels: Dict[Tuple[int, int, int], VoxelStats] = {}

        if self._origin is not None and self._origin.shape != (3,):
            raise ValueError("origin must be a 3-vector")
        if self._bounds is not None and self._bounds.shape != (2, 3):
            raise ValueError("bounds must be shaped (2, 3)")

    @property
    def origin(self) -> np.ndarray | None:
        """Return the world-space origin of the grid if initialised."""

        return self._origin

    def _ensure_origin(self, points: np.ndarray) -> None:
        if self._origin is not None:
            return

        if self._bounds is not None:
            self._origin = np.asarray(self._bounds[0], dtype=np.float64)
            return

        if points.size == 0:
            raise ValueError("Cannot infer origin from an empty point set")

        mins = points.min(axis=0)
        self._origin = np.floor(mins / self.voxel_size) * self.voxel_size

    def world_to_voxel(self, points: np.ndarray) -> np.ndarray:
        """Convert world coordinates to integer voxel indices."""

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be of shape (N, 3)")

        if points.size == 0:
            return np.empty((0, 3), dtype=np.int64)

        self._ensure_origin(points)
        rel = (points - self._origin) / self.voxel_size
        return np.floor(rel).astype(np.int64)

    def voxel_to_world(self, indices: np.ndarray) -> np.ndarray:
        """Convert voxel indices to world coordinates at voxel centres."""

        if self._origin is None:
            raise ValueError("origin not initialised")

        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("indices must be of shape (N, 3)")

        return indices * self.voxel_size + self._origin + self.voxel_size / 2.0

    def accumulate(self, voxel_idx: Tuple[int, int, int], logits: np.ndarray, weight: float = 1.0) -> None:
        """Add logits for a single voxel."""

        stats = self._voxels.get(voxel_idx)
        if stats is None:
            stats = VoxelStats(self.num_classes)
            self._voxels[voxel_idx] = stats
        stats.accumulate(np.asarray(logits, dtype=np.float64), weight)

    def bulk_accumulate(
        self,
        indices: Iterable[Tuple[int, int, int]],
        logits: np.ndarray,
        weights: Iterable[float] | None = None,
    ) -> None:
        """Accumulate logits for multiple voxels."""

        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError("logits must be shaped (N, num_classes)")

        if weights is None:
            weights_iter: Iterable[float] = (1.0 for _ in range(logits.shape[0]))
        else:
            weights_iter = weights

        for idx, vec, weight in zip(indices, logits, weights_iter):
            self.accumulate(tuple(idx), vec, float(weight))

    def items(self) -> Iterator[Tuple[Tuple[int, int, int], VoxelStats]]:
        """Iterate over stored voxels and their statistics."""

        return self._voxels.items()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._voxels)

    def get_logits_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return stacked voxel indices and logits arrays."""

        if not self._voxels:
            return (
                np.zeros((0, 3), dtype=np.int64),
                np.zeros((0, self.num_classes), dtype=np.float64),
            )

        indices = np.array(list(self._voxels.keys()), dtype=np.int64)
        logits = np.stack([voxel.logits for voxel in self._voxels.values()], axis=0)
        return indices, logits
