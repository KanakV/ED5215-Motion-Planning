import numpy as np
import heapq
from config.config import COLLISION_RADIUS

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
LOOKAHEAD = 8              # slightly larger horizon
WAIT_ALLOWED = True

GOAL_WEIGHT = 1.2          # bias toward runway
CONFLICT_PENALTY = 3       # soft penalty instead of hard block
INERTIA_BONUS = 0.3        # reward continuing direction


# ------------------------------------------------------------
# Fast squared distance
# ------------------------------------------------------------
def sq_dist(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ------------------------------------------------------------
# Improved Limited A*
# ------------------------------------------------------------
def limited_astar(grid, start, goal, blocked_positions, prev_pos):

    rows, cols = grid.shape

    open_list = []
    heapq.heappush(open_list, (0, 0, start, None))

    visited = set()
    parent = {}

    while open_list:

        f, g, (x, y), prev = heapq.heappop(open_list)

        if (x, y) in visited:
            continue

        visited.add((x, y))
        parent[(x, y)] = prev

        if g >= LOOKAHEAD or (x, y) == goal:
            path = []
            node = (x, y)
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]

        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        if WAIT_ALLOWED:
            moves.append((0,0))

        for dx, dy in moves:

            nx, ny = x + dx, y + dy

            if not (0 <= nx < rows and 0 <= ny < cols):
                continue

            if (nx, ny) in visited:
                continue

            # ------------------------------------------------
            # Soft conflict cost
            # ------------------------------------------------
            penalty = 0
            for bx, by in blocked_positions:
                if sq_dist((nx, ny), (bx, by)) < COLLISION_RADIUS**2:
                    penalty += CONFLICT_PENALTY

            # ------------------------------------------------
            # Goal bias
            # ------------------------------------------------
            h = GOAL_WEIGHT * manhattan((nx, ny), goal)

            # ------------------------------------------------
            # Inertia (reduce zig-zag)
            # ------------------------------------------------
            inertia = 0
            if prev_pos is not None:
                prev_dx = x - prev_pos[0]
                prev_dy = y - prev_pos[1]
                if (dx, dy) == (prev_dx, prev_dy):
                    inertia -= INERTIA_BONUS

            cost = g + 1 + penalty + h + inertia

            heapq.heappush(
                open_list,
                (cost, g + 1, (nx, ny), (x, y))
            )

    return [start]


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