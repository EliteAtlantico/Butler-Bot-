"""Occupancy grid accumulated from depth observations.

The grid is log-odds so that a single spurious return does not carve a
permanent obstacle into the map, and so that space the robot drives through
gets re-cleared. Cells the camera has never seen stay `unknown` and are
planned through optimistically -- the robot discovers the wall by looking at
it and replans, which is the behaviour you actually want from a robot that
only knows what its camera has told it.

`costmap()` is the bridge to `planning`: it turns occupancy into hard no-go
cells plus a soft cost that keeps paths off the walls.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover
    distance_transform_edt = None


@dataclass
class OccupancyGrid:
    resolution: float = 0.10
    origin: tuple[float, float] = (-6.0, -6.0)
    size: tuple[int, int] = (150, 130)      # cells in (x, y)
    hit: float = 0.85
    miss: float = 0.28
    clamp: float = 6.0
    # One hit must be enough to mark a cell, or the first plan is made against
    # an empty map; clearing then takes three consecutive misses.
    occupied_threshold: float = 0.40

    def __post_init__(self):
        self.logodds = np.zeros(self.size, np.float32)
        self.seen = np.zeros(self.size, bool)

    @classmethod
    def covering(cls, bounds, resolution: float = 0.10, **kw) -> "OccupancyGrid":
        """Size the grid to a world extent instead of a hard-coded default.

        The previous fixed +-6 m grid silently dropped every observation from
        a scene laid out anywhere else: to_cell returned out-of-range indices,
        inside() filtered them all away, and the map just stayed empty.
        """
        x0, y0, x1, y1 = bounds
        nx = max(int(np.ceil((x1 - x0) / resolution)), 1)
        ny = max(int(np.ceil((y1 - y0) / resolution)), 1)
        return cls(resolution=resolution, origin=(float(x0), float(y0)),
                   size=(nx, ny), **kw)

    # ------------------------------------------------------------- indexing
    def to_cell(self, xy) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, float))
        return np.floor((xy - self.origin) / self.resolution).astype(np.int64)

    def to_world(self, cell) -> np.ndarray:
        cell = np.atleast_2d(np.asarray(cell, float))
        return cell * self.resolution + self.origin + self.resolution / 2

    def inside(self, cell) -> np.ndarray:
        cell = np.atleast_2d(cell)
        return ((cell[:, 0] >= 0) & (cell[:, 0] < self.size[0]) &
                (cell[:, 1] >= 0) & (cell[:, 1] < self.size[1]))

    # ------------------------------------------------------------- updating
    def _cells_of(self, pts) -> np.ndarray:
        """World points -> unique in-bounds cells.

        Deduplicating matters for more than speed: thousands of pixels land on
        one surface, and letting each one vote turns a single observation into
        an unclearable wall.
        """
        if pts is None or len(pts) == 0:
            return np.zeros((0, 2), np.int64)
        cells = self.to_cell(np.asarray(pts)[:, :2])
        cells = cells[self.inside(cells)]
        if not len(cells):
            return cells
        # Dedup through a 1-D key: np.unique(axis=0) sorts a structured view
        # and is an order of magnitude slower at this point count.
        _, idx = np.unique(self._keys(cells), return_index=True)
        return cells[idx]

    def _keys(self, cells) -> np.ndarray:
        return cells[:, 0] * self.size[1] + cells[:, 1]

    def integrate(self, camera_xy, obstacle_xy, free_xy=None):
        """Raise obstacle cells, and clear the space the rays passed through."""
        origin = self.to_cell(camera_xy)[0]
        hits = self._cells_of(obstacle_xy)
        ends = [c for c in (hits, self._cells_of(free_xy)) if len(c)]
        if not ends:
            return
        free_keys = self._trace(origin, np.concatenate(ends, 0))
        hit_keys = self._keys(hits)
        # A cell seen as a surface this frame must not also be cleared by a ray
        # that grazed past it.
        if len(hit_keys) and len(free_keys):
            free_keys = free_keys[~np.isin(free_keys, hit_keys)]

        flat = self.logodds.reshape(-1)
        seen = self.seen.reshape(-1)
        flat[free_keys] -= self.miss
        flat[hit_keys] += self.hit
        np.clip(flat, -self.clamp, self.clamp, out=flat)
        seen[free_keys] = True
        seen[hit_keys] = True

    def _trace(self, origin, ends) -> np.ndarray:
        """Keys of the cells strictly between origin and each end, vectorised.

        Sampling along the segment rather than stepping Bresenham per ray: the
        aliasing is irrelevant once cells are deduplicated, and it turns
        thousands of Python loops into one array op. Keys come back rather
        than coordinate pairs so the dedup is a 1-D sort.
        """
        empty = np.zeros(0, np.int64)
        if len(ends) == 0:
            return empty
        deltas = ends - origin
        steps = int(np.abs(deltas).max()) + 1
        if steps < 2:
            return empty
        ts = np.linspace(0.0, 1.0, steps, endpoint=False)
        cx = np.floor(origin[0] + np.outer(deltas[:, 0], ts)).astype(np.int64)
        cy = np.floor(origin[1] + np.outer(deltas[:, 1], ts)).astype(np.int64)
        good = ((cx >= 0) & (cx < self.size[0]) &
                (cy >= 0) & (cy < self.size[1]))
        return np.unique(cx[good] * self.size[1] + cy[good])

    # -------------------------------------------------------------- queries
    @property
    def occupied(self) -> np.ndarray:
        return self.logodds > self.occupied_threshold

    @property
    def unknown(self) -> np.ndarray:
        return ~self.seen

    def costmap(self, robot_radius: float = 0.30, clearance: float = 0.45):
        """(blocked, cost) -- hard no-go cells, and a soft cost near them.

        The soft term keeps the path off the walls instead of shaving corners
        at exactly the inflation radius, which is where a two-wheeler with
        non-zero turning transients actually clips things.
        """
        occ = self.occupied
        if distance_transform_edt is None or not occ.any():
            return occ.copy(), np.zeros_like(self.logodds)
        dist = distance_transform_edt(~occ) * self.resolution
        blocked = dist < robot_radius
        soft = np.clip((clearance - dist) / max(clearance, 1e-6), 0.0, 1.0) ** 2
        return blocked, soft.astype(np.float32)


