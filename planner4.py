import heapq
import numpy as np
from copy import deepcopy
from itertools import count
from config.config import COLLISION_RADIUS


# ============================================================
# Manhattan Heuristic
# ============================================================

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ============================================================
# Low-Level Space-Time A*
# ============================================================

def space_time_astar(grid, start, goal, constraints, max_time=200):

    rows, cols = grid.shape
    open_list = []
    counter = count()

    # (f, g, tie_breaker, (x,y,t))
    heapq.heappush(
        open_list,
        (manhattan(start, goal), 0, next(counter), (start[0], start[1], 0))
    )

    parent = {}
    g_score = {(start[0], start[1], 0): 0}

    while open_list:

        f, g, _, (x, y, t) = heapq.heappop(open_list)

        if (x, y) == goal:
            # Reconstruct full path
            path = []
            node = (x, y, t)
            while node in parent:
                path.append((node[0], node[1]))
                node = parent[node]
            path.append(start)
            return path[::-1]

        if t >= max_time:
            continue

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(0,0)]:  # wait allowed

            nx, ny = x + dx, y + dy
            nt = t + 1

            if not (0 <= nx < rows and 0 <= ny < cols):
                continue

            # Vertex constraint
            if (nx, ny, nt) in constraints["vertex"]:
                continue

            # Edge constraint
            if (x, y, nx, ny, nt) in constraints["edge"]:
                continue

            # Separation constraint
            for (vx, vy, vt) in constraints["vertex"]:
                if vt == nt:
                    if np.linalg.norm(np.array((nx, ny)) - np.array((vx, vy))) < COLLISION_RADIUS:
                        break
            else:
                h = manhattan((nx, ny), goal)
                new_g = g + 1
                node = (nx, ny, nt)

                if node not in g_score or new_g < g_score[node]:
                    g_score[node] = new_g
                    heapq.heappush(
                        open_list,
                        (new_g + h, new_g, next(counter), node)
                    )
                    parent[node] = (x, y, t)

    return None


# ============================================================
# Conflict Detection
# ============================================================

def detect_conflict(paths):

    max_len = max(len(p) for p in paths.values())

    for t in range(max_len):

        positions = {}

        # Vertex conflicts
        for pid, path in paths.items():

            pos = path[t] if t < len(path) else path[-1]

            for other_pid, other_pos in positions.items():
                if np.linalg.norm(np.array(pos) - np.array(other_pos)) < COLLISION_RADIUS:
                    return {
                        "type": "vertex",
                        "time": t,
                        "a1": pid,
                        "a2": other_pid,
                        "pos": pos
                    }

            positions[pid] = pos

        # Edge conflicts
        for pid1 in paths:
            for pid2 in paths:
                if pid1 >= pid2:
                    continue

                if t+1 >= len(paths[pid1]) or t+1 >= len(paths[pid2]):
                    continue

                if paths[pid1][t] == paths[pid2][t+1] and \
                   paths[pid1][t+1] == paths[pid2][t]:
                    return {
                        "type": "edge",
                        "time": t+1,
                        "a1": pid1,
                        "a2": pid2
                    }

    return None


# ============================================================
# CBS Planner
# ============================================================

def cbs_planner(grid, active_planes, goal):

    if not active_planes:
        return {}

    counter = count()

    # Root node
    root = {
        "constraints": {},
        "paths": {},
        "cost": 0
    }

    # Initial paths
    for plane in active_planes:
        pid = plane["id"]

        root["constraints"][pid] = {
            "vertex": set(),
            "edge": set()
        }

        path = space_time_astar(
            grid,
            plane["pos"],
            goal,
            root["constraints"][pid]
        )

        if path is None:
            return {}

        root["paths"][pid] = path

    root["cost"] = sum(len(p) for p in root["paths"].values())

    open_list = []
    heapq.heappush(open_list, (root["cost"], next(counter), root))

    while open_list:

        _, _, node = heapq.heappop(open_list)

        conflict = detect_conflict(node["paths"])

        if conflict is None:
            # SUCCESS → return next actions only
            actions = {}
            for pid, path in node["paths"].items():
                if len(path) > 1:
                    actions[pid] = path[1]
                else:
                    actions[pid] = path[0]
            return actions

        # Split on conflict
        for agent in [conflict["a1"], conflict["a2"]]:

            new_node = deepcopy(node)

            if conflict["type"] == "vertex":
                new_node["constraints"][agent]["vertex"].add(
                    (conflict["pos"][0], conflict["pos"][1], conflict["time"])
                )

            else:  # edge conflict
                path = node["paths"][agent]
                x1, y1 = path[conflict["time"] - 1]
                x2, y2 = path[conflict["time"]]
                new_node["constraints"][agent]["edge"].add(
                    (x1, y1, x2, y2, conflict["time"])
                )

            plane = next(p for p in active_planes if p["id"] == agent)

            new_path = space_time_astar(
                grid,
                plane["pos"],
                goal,
                new_node["constraints"][agent]
            )

            if new_path is None:
                continue

            new_node["paths"][agent] = new_path
            new_node["cost"] = sum(len(p) for p in new_node["paths"].values())

            heapq.heappush(
                open_list,
                (new_node["cost"], next(counter), new_node)
            )

    return {}