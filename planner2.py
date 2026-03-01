
import numpy as np
import heapq
from config.config import COLLISION_RADIUS
# ------------------------------------------------------------
# Manhattan Distance Heuristic
# ------------------------------------------------------------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ------------------------------------------------------------
# A* in Space-Time (No Wait Action)
# ------------------------------------------------------------

def space_time_astar(grid, start, goal, reservations, max_time=200):
    """
    A* in space-time.
    State: (x, y, t)
    """

    rows, cols = grid.shape

    open_list = []
    heapq.heappush(
        open_list,
        (manhattan(start, goal), 0, (start[0], start[1], 0), None)
    )

    visited = set()
    parent = {}

    while open_list:

        f, g, (x, y, t), prev = heapq.heappop(open_list)

        if (x, y, t) in visited:
            continue

        visited.add((x, y, t))
        parent[(x, y, t)] = prev

        # Goal reached
        if (x, y) == goal:
            path = []
            node = (x, y, t)
            while node is not None:
                path.append((node[0], node[1]))
                node = parent[node]
            return path[::-1]

        if t >= max_time:
            continue

        # 4-connected moves (NO WAIT)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:

            nx, ny = x + dx, y + dy
            nt = t + 1

            if not (0 <= nx < rows and 0 <= ny < cols):
                continue

            # Vertex reservation check
            if (nx, ny, nt) in reservations["vertices"]:
                continue

            # Edge swap reservation
            if (nx, ny, x, y, nt) in reservations["edges"]:
                continue

            # Separation check
            conflict = False
            for (rx, ry, rt) in reservations["vertices"]:
                if rt == nt:
                    if np.linalg.norm(np.array((nx, ny)) - np.array((rx, ry))) < COLLISION_RADIUS:
                        conflict = True
                        break
            if conflict:
                continue

            h = manhattan((nx, ny), goal)
            heapq.heappush(
                open_list,
                (g + 1 + h, g + 1, (nx, ny, nt), (x, y, t))
            )

    return None


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