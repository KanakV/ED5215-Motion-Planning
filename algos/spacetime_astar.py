"""
spacetime_astar.py
Unified space-time A* that supports:
  - Constraint sets: (cell, t) pairs the agent must avoid
  - Reservation tables: vertex, edge, and COLLISION_RADIUS separation checks
  - Spawn-edge avoidance bias for the first few timesteps
  - best_g pruning for efficiency
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config.config import COLLISION_RADIUS, SPAWN_BIAS_STEPS
from algos.algo_helpers import manhattan

# Type aliases
Pos = Tuple[int, int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _heuristic(pos: Pos, goal: Pos) -> int:
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def _grid_neighbors(pos: Pos, grid_size: int) -> List[Pos]:
    """4-connected neighbors that stay within a square grid of side grid_size."""
    x, y = pos
    candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    return [(nx, ny) for nx, ny in candidates if 0 <= nx < grid_size and 0 <= ny < grid_size]


def _array_neighbors(pos: Pos, rows: int, cols: int) -> List[Pos]:
    """4-connected neighbors that stay within a rectangular grid (rows × cols)."""
    x, y = pos
    candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    return [(nx, ny) for nx, ny in candidates if 0 <= nx < rows and 0 <= ny < cols]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spacetime_astar(
    start: Pos,
    goal: Pos,
    *,
    # --- grid description (provide one) ---
    grid_size: Optional[int] = None,          # square grid  (original file-1 style)
    grid: Optional[np.ndarray] = None,        # numpy array  (original file-2 style)
    # --- constraint / reservation tables ---
    constraints: Optional[Set[Tuple[Pos, int]]] = None,   # {(cell, t), …}
    reservations: Optional[Dict] = None,                   # {vertices, edges}
    # --- tuning ---
    max_t: int = 300,
    spawn_bias_steps: int = SPAWN_BIAS_STEPS,
    spawn_bias_weight: float = 1e-3,
) -> Optional[List[Pos]]:
    """
    Shortest path from *start* to *goal* in space-time.

    Grid specification
    ------------------
    Provide either ``grid_size`` (square grid, side length) **or** ``grid``
    (a 2-D numpy array where shape gives rows × cols).  If both are given,
    ``grid`` takes precedence.

    Constraint / reservation tables
    --------------------------------
    ``constraints``
        A set of ``(cell, t)`` tuples.  The agent must not occupy ``cell``
        at time ``t``.

    ``reservations``
        A dict with optional keys:

        * ``"vertices"`` – set of ``(x, y, t)`` triples the agent must avoid.
        * ``"edges"``    – set of ``(nx, ny, x, y, t)`` tuples that forbid the
          agent moving from ``(x,y)`` to ``(nx,ny)`` at time ``t``.

        Vertex entries are also used for separation checks: any move that
        brings the agent within ``COLLISION_RADIUS`` of a reserved vertex at
        the same timestep is rejected.

    Spawn-edge bias
    ---------------
    For the first ``spawn_bias_steps`` timesteps, moves onto the grid border
    receive a small additive penalty (``spawn_bias_weight``) in the priority,
    discouraging newly spawned agents from hugging edges.

    Returns
    -------
    A list of ``Pos`` from *start* to *goal* (inclusive), or ``None`` if no
    path exists within *max_t* steps.
    """
    # ------------------------------------------------------------------ setup
    if grid is not None:
        rows, cols = grid.shape
        def neighbors(pos: Pos) -> List[Pos]:
            return _array_neighbors(pos, rows, cols)
        def on_border(pos: Pos) -> bool:
            x, y = pos
            return x == 0 or y == 0 or x == rows - 1 or y == cols - 1
    elif grid_size is not None:
        def neighbors(pos: Pos) -> List[Pos]:
            return _grid_neighbors(pos, grid_size)
        def on_border(pos: Pos) -> bool:
            x, y = pos
            return x == 0 or y == 0 or x == grid_size - 1 or y == grid_size - 1
    else:
        raise ValueError("Provide either `grid_size` or `grid`.")

    _constraints  = constraints  or set()
    _reservations = reservations or {}
    _v_res: Set   = _reservations.get("vertices", set())
    _e_res: Set   = _reservations.get("edges",    set())

    # ------------------------------------------------------------ A* machinery
    counter   = 0
    open_heap: list = [(_heuristic(start, goal), counter, 0, start)]
    came_from: Dict[Tuple[Pos, int], Optional[Tuple[Pos, int]]] = {(start, 0): None}
    best_g:    Dict[Tuple[Pos, int], int]                        = {(start, 0): 0}

    while open_heap:
        _f, _ctr, g, pos = heapq.heappop(open_heap)
        t = g

        # Prune stale heap entries
        if best_g.get((pos, t), float("inf")) < g:
            continue

        # ---------------------------------------------------------- goal check
        if pos == goal:
            path, state = [], (pos, t)
            while state is not None:
                path.append(state[0])
                state = came_from[state]
            path.reverse()
            return path

        if t >= max_t:
            continue

        # ---------------------------------------------------- expand neighbors
        for npos in neighbors(pos):
            nx, ny = npos
            nt = t + 1
            ng = g + 1

            # --- constraint set check (file-1 style) ---
            if (npos, nt) in _constraints:
                continue

            # --- reservation table checks (file-2 style) ---
            if (nx, ny, nt) in _v_res:
                continue
            if (nx, ny, pos[0], pos[1], nt) in _e_res:
                continue

            # Separation / collision-radius check against all reserved vertices
            # at the same timestep
            conflict = any(
                rt == nt and
                np.linalg.norm(np.array((nx, ny)) - np.array((rx, ry))) < COLLISION_RADIUS
                for rx, ry, rt in _v_res
            )
            if conflict:
                continue

            # --- best_g pruning ---
            key = (npos, nt)
            if best_g.get(key, float("inf")) <= ng:
                continue
            best_g[key]    = ng
            came_from[key] = (pos, t)

            # --- spawn-edge bias ---
            bias = spawn_bias_weight if (t < spawn_bias_steps and on_border(npos)) else 0

            counter += 1
            heapq.heappush(
                open_heap,
                (ng + _heuristic(npos, goal) + bias, counter, ng, npos),
            )

    return None