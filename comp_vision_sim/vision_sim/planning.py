"""Path planning over an occupancy costmap: A* plus string-pulling.

Takes the `blocked`/`soft` arrays that `occupancy.OccupancyGrid.costmap()`
produces and returns a route in cell coordinates. Nothing here knows about
cameras or robots -- it is grid in, grid out.
"""
from __future__ import annotations

import heapq

import numpy as np


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
