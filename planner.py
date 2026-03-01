import numpy as np
import heapq
from config.config import COLLISION_RADIUS

def simple_planner(grid, start, goal):
    path = []
    x, y = start
    gx, gy = goal

    while (x, y) != (gx, gy):
        if x < gx:
            x += 1
        elif x > gx:
            x -= 1
        elif y < gy:
            y += 1
        elif y > gy:
            y -= 1

        path.append((x, y))

    return path 


def centralized_planner(grid, active_planes, goal):

    actions = {}
    reserved = set()
    current_positions = {p["id"]: p["pos"] for p in active_planes}

    for plane in sorted(active_planes, key=lambda x: x["id"]):

        x, y = plane["pos"]

        # Possible moves (4-connected + wait)
        candidates = [
            (x+1, y),
            (x-1, y),
            (x, y+1),
            (x, y-1),
            (x, y)
        ]

        # Keep inside bounds
        candidates = [
            c for c in candidates
            if 0 <= c[0] < grid.shape[0] and 0 <= c[1] < grid.shape[1]
        ]

        # Sort by distance to goal
        candidates.sort(key=lambda c: np.linalg.norm(np.array(c) - np.array(goal)))

        for c in candidates:

            # Avoid reserved cell
            if c in reserved:
                continue

            # Avoid head-on swap
            swap = False
            for other_id, other_pos in current_positions.items():
                if other_id == plane["id"]:
                    continue
                if other_pos == c and actions.get(other_id) == plane["pos"]:
                    swap = True
                    break

            if swap:
                continue

            actions[plane["id"]] = c
            reserved.add(c)
            break

    return actions



# Anup ... Do not modify anythin below


