"""Occupancy grid built from depth, and an A* planner over it.

The grid is log-odds so that a single spurious return does not carve a
permanent obstacle into the map, and so that space the robot drives through
gets re-cleared. Cells the camera has never seen stay `unknown` and are
planned through optimistically -- the robot discovers the wall by looking at
it and replans, which is the behaviour you actually want from a robot that
only knows what its camera has told it.
"""
from __future__ import annotations

import heapq
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


_NEIGHBOURS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               if (dx, dy) != (0, 0)]


def astar(blocked: np.ndarray, start, goal, soft=None, soft_weight: float = 2.5,
          unknown_penalty: np.ndarray | None = None):
    """8-connected A* in cell space. Returns a list of cells, or None."""
    nx, ny = blocked.shape
    start, goal = tuple(int(v) for v in start), tuple(int(v) for v in goal)
    if not (0 <= goal[0] < nx and 0 <= goal[1] < ny):
        return None
    if blocked[goal]:
        goal = _nearest_free(blocked, goal)
        if goal is None:
            return None

    def h(c):
        return float(np.hypot(c[0] - goal[0], c[1] - goal[1]))

    open_heap = [(h(start), 0.0, start)]
    came: dict[tuple, tuple] = {}
    best = {start: 0.0}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if g > best.get(cur, np.inf):
            continue
        for dx, dy in _NEIGHBOURS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny) or blocked[nxt]:
                continue
            step = 1.4142 if dx and dy else 1.0
            if soft is not None:
                step *= 1.0 + soft_weight * float(soft[nxt])
            if unknown_penalty is not None:
                step += float(unknown_penalty[nxt])
            ng = g + step
            if ng < best.get(nxt, np.inf):
                best[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None


def _nearest_free(blocked, cell, max_radius: int = 25):
    """Snap a goal that landed inside inflation out to the closest open cell."""
    nx, ny = blocked.shape
    for r in range(1, max_radius):
        for dx in range(-r, r + 1):
            for dy in (-r, r) if abs(dx) != r else range(-r, r + 1):
                c = (cell[0] + dx, cell[1] + dy)
                if 0 <= c[0] < nx and 0 <= c[1] < ny and not blocked[c]:
                    return c
    return None


def line_of_sight(blocked: np.ndarray, a, b) -> bool:
    x0, y0 = int(a[0]), int(a[1])
    x1, y1 = int(b[0]), int(b[1])
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if blocked[x0, y0]:
            return False
        if (x0, y0) == (x1, y1):
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def shortcut(blocked: np.ndarray, path):
    """String-pulling: drop waypoints the robot can drive straight past."""
    if not path:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_of_sight(blocked, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out
