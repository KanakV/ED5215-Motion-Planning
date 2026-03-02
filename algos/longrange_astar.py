from algos.algo_helpers import manhattan
from algos.limited_astar import limited_astar
# ------------------------------------------------------------
# Optimized LRA* Planner
# ------------------------------------------------------------
def lra_planner(grid, active_planes, goal):

    actions = {}

    if not active_planes:
        return actions

    # Priority: closer planes move first
    sorted_planes = sorted(
        active_planes,
        key=lambda p: manhattan(p["pos"], goal)
    )

    reserved_positions = []

    for plane in sorted_planes:

        pid = plane["id"]
        start = plane["pos"]
        prev = plane.get("prev_pos", None)

        path = limited_astar(
            grid,
            start,
            goal,
            reserved_positions,
            prev
        )

        next_pos = path[1] if len(path) > 1 else start

        actions[pid] = next_pos
        reserved_positions.append(next_pos)

    return actions