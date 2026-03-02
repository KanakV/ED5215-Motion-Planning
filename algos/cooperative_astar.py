from algos.algo_helpers import manhattan
from algos.spacetime_astar import space_time_astar

# ------------------------------------------------------------
# Cooperative A* Planner
# ------------------------------------------------------------
def cooperative_planner(grid, active_planes, goal):

    actions = {}

    # Reservation structure
    reservations = {
        "vertices": set(),
        "edges": set()
    }

    # -------------------------------------------------
    #  Distance-based priority
    # -------------------------------------------------
    sorted_planes = sorted(
        active_planes,
        key=lambda p: manhattan(p["pos"], goal)
    )

    # -------------------------------------------------
    #  Plan sequentially
    # -------------------------------------------------
    for plane in sorted_planes:

        pid = plane["id"]
        start = plane["pos"]

        path = space_time_astar(
            grid,
            start,
            goal,
            reservations
        )

        if path is None or len(path) < 2:
            actions[pid] = start
            continue

        next_pos = path[1]
        actions[pid] = next_pos

        # -------------------------------------------------
        #  Reserve full path in space-time
        # -------------------------------------------------
        for t in range(len(path)):
            x, y = path[t]
            reservations["vertices"].add((x, y, t))

            if t > 0:
                px, py = path[t - 1]
                reservations["edges"].add((px, py, x, y, t))

    return actions