import numpy as np
import heapq
from config.config import COLLISION_RADIUS
from config.config import LOOKAHEAD, WAIT_ALLOWED, GOAL_WEIGHT, CONFLICT_PENALTY, INERTIA_BONUS
from algos.algo_helpers import manhattan, sq_dist

# ------------------------------------------------------------
# Center-pull bias (mirrors spacetime_astar edge-avoidance)
#
# spacetime_astar adds +epsilon when moving ONTO a border cell.
# Here we add +CENTER_BIAS * dist_to_center so border cells are
# naturally more expensive, pulling agents toward the interior.
#
# Only applied for the first CENTER_BIAS_STEPS steps of the path
# (same "shortly after spawn" logic as in spacetime_astar).
# ------------------------------------------------------------
CENTER_BIAS_WEIGHT = 1e-3   # matches epsilon in spacetime_astar
CENTER_BIAS_STEPS  = 6      # matches spawn_bias_steps in spacetime_astar


def limited_astar(grid, start, goal, blocked_positions, prev_pos):
    rows, cols = grid.shape
    center = ((rows - 1) / 2.0, (cols - 1) / 2.0)

    open_list = []
    heapq.heappush(open_list, (0, 0, start, None))
    visited = set()
    parent  = {}

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

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if WAIT_ALLOWED:
            moves.append((0, 0))

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
                if sq_dist((nx, ny), (bx, by)) < COLLISION_RADIUS ** 2:
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

            # ------------------------------------------------
            # Center-pull bias — mirrors spacetime_astar's
            # edge-avoidance; only active shortly after spawn.
            # Euclidean distance from candidate cell to center,
            # weighted small so it never overrides real costs.
            # ------------------------------------------------
            center_bias = 0
            if g < CENTER_BIAS_STEPS:
                dist_to_center = ((nx - center[0]) ** 2 + (ny - center[1]) ** 2) ** 0.5
                center_bias = CENTER_BIAS_WEIGHT * dist_to_center

            cost = g + 1 + penalty + h + inertia + center_bias
            heapq.heappush(
                open_list,
                (cost, g + 1, (nx, ny), (x, y))
            )

    return [start]