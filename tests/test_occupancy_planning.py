from __future__ import annotations

import numpy as np
import pytest

from vision_sim.occupancy import OccupancyGrid
from vision_sim import occupancy, planning


def grid(**kwargs):
    return OccupancyGrid(resolution=1.0, origin=(0.0, 0.0), size=(8, 6), **kwargs)


def test_grid_initial_state_is_unknown_and_unoccupied():
    g = grid()
    assert g.logodds.shape == (8, 6)
    assert g.logodds.dtype == np.float32
    assert g.unknown.all()
    assert not g.occupied.any()


def test_world_cell_round_trip_uses_cell_centres():
    g = grid()
    cells = g.to_cell([[0.0, 0.0], [1.99, 2.01], [-0.01, 0]])
    assert np.array_equal(cells, [[0, 0], [1, 2], [-1, 0]])
    assert np.array_equal(g.to_world([[0, 0], [1, 2]]), [[0.5, 0.5], [1.5, 2.5]])


def test_inside_vectorized_boundaries():
    g = grid()
    assert np.array_equal(g.inside([[-1, 0], [0, 0], [7, 5], [8, 5]]),
                          [False, True, True, False])


def test_cells_of_handles_none_empty_deduplicates_and_clips():
    g = grid()
    assert g._cells_of(None).shape == (0, 2)
    assert g._cells_of([]).shape == (0, 2)
    cells = g._cells_of([[1.1, 2.2], [1.9, 2.8], [-2, 0], [99, 99]])
    assert np.array_equal(cells, [[1, 2]])


def test_keys_follow_row_major_grid_shape():
    g = grid()
    assert np.array_equal(g._keys(np.array([[0, 0], [1, 2], [7, 5]])), [0, 8, 47])


def test_trace_empty_and_same_cell_are_empty():
    g = grid()
    assert g._trace(np.array([1, 1]), np.empty((0, 2), int)).size == 0
    assert g._trace(np.array([1, 1]), np.array([[1, 1]])).size == 0


def test_trace_contains_origin_and_intermediate_but_not_endpoint():
    g = grid()
    keys = g._trace(np.array([0, 0]), np.array([[4, 0]]))
    assert np.array_equal(keys, [0, 6, 12, 18])
    assert 24 not in keys


def test_integrate_no_returns_is_noop():
    g = grid()
    g.integrate((0, 0), [], [])
    assert not g.seen.any()
    assert not g.logodds.any()


def test_integrate_marks_hit_and_clears_ray_without_double_voting():
    g = grid()
    g.integrate((0.1, 0.1), [[4.1, 0.1]], [[4.1, 0.1]])
    assert g.logodds[4, 0] == pytest.approx(g.hit)
    assert g.logodds[0, 0] == pytest.approx(-g.miss)
    assert g.seen[0, 0] and g.seen[4, 0]


def test_integrate_deduplicates_pixel_votes():
    g = grid()
    g.integrate((0, 0), np.repeat([[2.1, 2.1]], 100, axis=0))
    assert g.logodds[2, 2] == pytest.approx(g.hit)


def test_integrate_clamps_positive_and_negative_logodds():
    g = grid(clamp=1.0)
    for _ in range(10):
        g.integrate((0, 0), [[2.1, 0.1]])
    assert g.logodds[2, 0] == 1.0
    assert g.logodds[0, 0] == -1.0


def test_occupied_threshold_is_strictly_greater():
    g = grid(occupied_threshold=0.4)
    g.logodds[1, 1] = 0.4
    g.logodds[2, 2] = 0.401
    assert not g.occupied[1, 1]
    assert g.occupied[2, 2]


def test_costmap_without_occupied_cells_has_zero_cost():
    g = grid()
    blocked, soft = g.costmap()
    assert not blocked.any()
    assert not soft.any()
    assert blocked is not g.occupied


def test_costmap_fallback_without_scipy(monkeypatch):
    g = grid()
    g.logodds[2, 2] = 2
    monkeypatch.setattr(occupancy, "distance_transform_edt", None)
    blocked, soft = g.costmap()
    assert np.array_equal(blocked, g.occupied)
    assert not soft.any()


def test_costmap_inflates_obstacles_and_adds_soft_clearance():
    g = OccupancyGrid(resolution=0.1, origin=(0, 0), size=(20, 20))
    g.logodds[10, 10] = 1
    blocked, soft = g.costmap(robot_radius=0.21, clearance=0.5)
    assert blocked[10, 10] and blocked[11, 10] and blocked[12, 10]
    assert not blocked[13, 10]
    assert soft[10, 10] == pytest.approx(1.0)
    assert 0 < soft[13, 10] < 1
    assert soft.dtype == np.float32


def test_astar_straight_cardinal_path():
    path = planning.astar(np.zeros((5, 5), bool), (0, 0), (4, 0))
    assert path[0] == (0, 0) and path[-1] == (4, 0)
    assert len(path) == 5


def test_astar_prefers_diagonal_moves():
    path = planning.astar(np.zeros((5, 5), bool), (0, 0), (4, 4))
    assert path == [(i, i) for i in range(5)]


def test_astar_routes_around_wall_gap():
    blocked = np.zeros((7, 7), bool)
    blocked[3, :] = True
    blocked[3, 5] = False
    path = planning.astar(blocked, (1, 1), (5, 1))
    assert path is not None
    assert (3, 5) in path
    assert all(not blocked[p] for p in path)


def test_astar_returns_none_for_out_of_bounds_goal():
    blocked = np.zeros((3, 3), bool)
    assert planning.astar(blocked, (0, 0), (3, 0)) is None
    assert planning.astar(blocked, (0, 0), (-1, 0)) is None


def test_astar_snaps_blocked_goal_to_nearest_free():
    blocked = np.zeros((5, 5), bool)
    blocked[2, 2] = True
    path = planning.astar(blocked, (0, 0), (2, 2))
    assert path is not None and path[-1] != (2, 2)
    assert not blocked[path[-1]]


def test_astar_returns_none_when_goal_has_no_free_cell_in_radius():
    blocked = np.ones((55, 55), bool)
    blocked[0, 0] = False
    assert planning.astar(blocked, (0, 0), (27, 27)) is None


def test_astar_returns_none_for_disconnected_regions():
    blocked = np.zeros((5, 5), bool)
    blocked[2, :] = True
    assert planning.astar(blocked, (0, 0), (4, 4)) is None


def test_astar_soft_cost_avoids_expensive_cells():
    blocked = np.zeros((7, 5), bool)
    soft = np.zeros_like(blocked, float)
    soft[1:6, 2] = 100
    path = planning.astar(blocked, (0, 2), (6, 2), soft=soft)
    assert any(p[1] != 2 for p in path[1:-1])


def test_astar_unknown_penalty_avoids_unknown_cells():
    blocked = np.zeros((7, 5), bool)
    penalty = np.zeros_like(blocked, float)
    penalty[1:6, 2] = 10
    path = planning.astar(blocked, (0, 2), (6, 2), unknown_penalty=penalty)
    assert any(p[1] != 2 for p in path[1:-1])


def test_nearest_free_respects_bounds_and_limit():
    blocked = np.ones((4, 4), bool)
    blocked[0, 1] = False
    assert planning._nearest_free(blocked, (0, 0)) == (0, 1)
    assert planning._nearest_free(np.ones((4, 4), bool), (0, 0), max_radius=2) is None


@pytest.mark.parametrize(("a", "b"), [((0, 0), (4, 0)), ((4, 0), (0, 0)),
                                         ((0, 0), (4, 4)), ((0, 4), (4, 0))])
def test_line_of_sight_clear_in_all_directions(a, b):
    assert planning.line_of_sight(np.zeros((5, 5), bool), a, b)


def test_line_of_sight_includes_both_endpoints_and_middle():
    for obstacle in [(0, 0), (2, 2), (4, 4)]:
        blocked = np.zeros((5, 5), bool)
        blocked[obstacle] = True
        assert not planning.line_of_sight(blocked, (0, 0), (4, 4))


def test_shortcut_preserves_empty_singleton_and_direct_path():
    blocked = np.zeros((5, 5), bool)
    assert planning.shortcut(blocked, []) == []
    assert planning.shortcut(blocked, [(1, 1)]) == [(1, 1)]
    assert planning.shortcut(blocked, [(0, 0), (1, 0), (2, 0)]) == [(0, 0), (2, 0)]


def test_shortcut_keeps_required_corner():
    blocked = np.zeros((5, 5), bool)
    blocked[2, 2] = True
    path = [(0, 2), (1, 1), (2, 0), (3, 1), (4, 2)]
    short = planning.shortcut(blocked, path)
    assert short[0] == path[0] and short[-1] == path[-1]
    assert len(short) > 2
