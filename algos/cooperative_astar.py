"""
Cooperative A* Planner
========================
Sequential (prioritised) multi-agent planner.

Agents are sorted by Manhattan distance to goal (closest first) and planned
one at a time.  Each planned path is inserted into a shared reservation table
so later agents route around earlier ones.

Low-level planner: spacetime_astar(..., grid=grid, reservations=...).
"""

from typing import Dict, List, Tuple

from algos.spacetime_astar import spacetime_astar
from algos.algo_helpers import manhattan

Pos = Tuple[int, int]


def cooperative_planner(
    grid,
    active_planes: List[Dict],
    goal: Pos,
) -> Dict[int, Pos]:
    """
    Plan one step for every active plane and return {agent_id: next_pos}.

    Uses a shared reservation table (vertices + edges) built up as each
    agent's full space-time path is committed in priority order.
    """
    actions: Dict[int, Pos] = {}

    # Shared reservation table (file-2 style)
    reservations = {
        "vertices": set(),   # (x, y, t)
        "edges":    set(),   # (nx, ny, px, py, t)  — swap conflicts
    }

    # Sort closest-to-goal first so they get right-of-way
    sorted_planes = sorted(active_planes, key=lambda p: manhattan(p["pos"], goal))

    for plane in sorted_planes:
        pid   = plane["id"]
        start = plane["pos"]

        path = spacetime_astar(
            start,
            goal,
            grid=grid,
            reservations=reservations,
        )

        if path is None or len(path) < 2:
            # No path found — agent stays put this step
            actions[pid] = start
            continue

        actions[pid] = path[1]

        # Commit the full path into the reservation table so subsequent
        # agents route around this one
        for t, (x, y) in enumerate(path):
            reservations["vertices"].add((x, y, t))
            if t > 0:
                px, py = path[t - 1]
                reservations["edges"].add((px, py, x, y, t))

    return actions