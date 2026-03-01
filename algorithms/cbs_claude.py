"""
Conflict-Based Search (CBS) Planner — Diagnostic Version
=========================================================
Fixes applied vs previous version:
  1. Vertex constraints: single cell only (removed _cells_within_radius disc)
  2. _occupancy_constraints_at_t0: single cell only, not a disc
  3. _occupancy_constraints_at_t0 removed from CT child replanning (root only)
  4. Edge constraint bug fixed: no longer overwrites with vertex constraint
"""

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

from config.config import PLANE_RADIUS, WARNING_RADIUS, MAX_NODES

import logging

logging.basicConfig(
    filename="cbs_debug.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(message)s",
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Pos        = Tuple[int, int]
Constraint = Tuple[int, Pos, int]

_BUBBLE_R = WARNING_RADIUS / 2.0

# Step counter for readable logs
_step = 0

# Collision tracer state
_last_actions:   dict = {}
_last_positions: dict = {}
_last_conflicts: list = []


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _cells_within_radius(center: Pos, radius: float, grid_size: int) -> List[Pos]:
    cr, cc = center
    r_ceil = math.ceil(radius)
    cells  = []
    for dr in range(-r_ceil, r_ceil + 1):
        for dc in range(-r_ceil, r_ceil + 1):
            if math.sqrt(dr * dr + dc * dc) <= radius:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    cells.append((nr, nc))
    return cells


def _neighbors_no_wait(pos: Pos, grid_size: int) -> List[Pos]:
    r, c = pos
    return [
        (nr, nc)
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        if 0 <= nr < grid_size and 0 <= nc < grid_size
    ]


def _heuristic(pos: Pos, goal: Pos) -> int:
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


# ---------------------------------------------------------------------------
# Space-Time A*
# ---------------------------------------------------------------------------

def _spacetime_astar(
    start:            Pos,
    goal:             Pos,
    grid_size:        int,
    constraints:      Set[Tuple[Pos, int]],
    edge_constraints: Set[Tuple[Pos, Pos, int]],
    max_t:            int = 300,
) -> Optional[List[Pos]]:
    h0        = _heuristic(start, goal)
    open_heap = [(h0, 0, start, 0)]
    came_from: Dict[Tuple[Pos, int], Optional[Tuple[Pos, int]]] = {(start, 0): None}
    best_g:    Dict[Tuple[Pos, int], int]                        = {(start, 0): 0}

    while open_heap:
        f, g, pos, t = heapq.heappop(open_heap)

        if best_g.get((pos, t), float("inf")) < g:
            continue

        if pos == goal:
            path, state = [], (pos, t)
            while state is not None:
                path.append(state[0])
                state = came_from[state]
            path.reverse()
            return path

        if t >= max_t:
            continue

        nt = t + 1
        for npos in _neighbors_no_wait(pos, grid_size):
            if (npos, nt) in constraints:
                continue
            if (pos, npos, t) in edge_constraints:
                continue

            ng  = g + 1
            key = (npos, nt)
            if best_g.get(key, float("inf")) <= ng:
                continue

            best_g[key]    = ng
            came_from[key] = (pos, t)
            heapq.heappush(open_heap, (ng + _heuristic(npos, goal), ng, npos, nt))

    return None


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

def _build_constraint_sets(
    raw:      List[Constraint],
    agent_id: int,
) -> Tuple[Set[Tuple[Pos, int]], Set[Tuple[Pos, Pos, int]]]:
    vertex: Set[Tuple[Pos, int]]      = set()
    edge:   Set[Tuple[Pos, Pos, int]] = set()
    for (aid, loc, t) in raw:
        if aid != agent_id:
            continue
        if isinstance(loc[0], tuple):
            edge.add((loc[0], loc[1], t))
        else:
            vertex.add((loc, t))
    return vertex, edge


def _occupancy_constraints_at_t0(
    agents:    List[Dict],
    agent_id:  int,
) -> Set[Tuple[Pos, int]]:
    """
    FIX: Block only the exact cell each other agent occupies at t=0.
    Previously used _cells_within_radius which added ~21 cells per agent,
    exploding the constraint sets and exhausting the CT budget immediately.
    """
    blocked: Set[Tuple[Pos, int]] = set()
    for other in agents:
        if other["id"] == agent_id:
            continue
        blocked.add((other["pos"], 0))
    return blocked


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def _find_first_conflict(paths: Dict[int, List[Pos]]) -> Optional[Tuple]:
    agent_ids = list(paths.keys())
    max_len   = max((len(p) for p in paths.values()), default=0)
    best      = None

    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            a, b   = agent_ids[i], agent_ids[j]
            pa, pb = paths[a], paths[b]

            for t in range(max_len):
                pos_a = pa[min(t, len(pa) - 1)]
                pos_b = pb[min(t, len(pb) - 1)]

                dist = math.sqrt(
                    (pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2
                )

                if dist < WARNING_RADIUS:
                    if best is None or t < best[3]:
                        best = (a, b, pos_a, t, "vertex")
                    break

                if t + 1 < max_len:
                    npos_a = pa[min(t + 1, len(pa) - 1)]
                    npos_b = pb[min(t + 1, len(pb) - 1)]
                    if pos_a == npos_b and pos_b == npos_a:
                        if best is None or t < best[3]:
                            best = (a, b, (pos_a, pos_b), t, "edge")
                        break

    return best


def _find_all_conflicts(paths: Dict[int, List[Pos]]) -> List[Tuple]:
    agent_ids = list(paths.keys())
    max_len   = max((len(p) for p in paths.values()), default=0)
    conflicts = []
    seen      = set()

    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            a, b   = agent_ids[i], agent_ids[j]
            pa, pb = paths[a], paths[b]

            for t in range(max_len):
                pos_a = pa[min(t, len(pa) - 1)]
                pos_b = pb[min(t, len(pb) - 1)]

                dist = math.sqrt(
                    (pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2
                )

                pair = (min(a, b), max(a, b))

                if dist < WARNING_RADIUS and pair not in seen:
                    seen.add(pair)
                    conflicts.append((a, b, pos_a, t, "vertex", dist))
                    break

                if t + 1 < max_len:
                    npos_a = pa[min(t + 1, len(pa) - 1)]
                    npos_b = pb[min(t + 1, len(pb) - 1)]
                    if pos_a == npos_b and pos_b == npos_a and pair not in seen:
                        seen.add(pair)
                        conflicts.append((a, b, (pos_a, pos_b), t, "edge", 0.0))
                        break

    return conflicts


# ---------------------------------------------------------------------------
# CBS Constraint Tree Node
# ---------------------------------------------------------------------------

class CTNode:
    __slots__ = ("constraints", "paths", "cost")

    def __init__(self, constraints: List[Constraint], paths: Dict[int, List[Pos]]):
        self.constraints = constraints
        self.paths       = paths
        self.cost        = sum(len(p) - 1 for p in paths.values())

    def __lt__(self, other: "CTNode"):
        return self.cost < other.cost


# ---------------------------------------------------------------------------
# High-level CBS
# ---------------------------------------------------------------------------

def _cbs(
    agents:    List[Dict],
    goal:      Pos,
    grid_size: int,
    max_nodes: int = MAX_NODES,
) -> Tuple[Dict[int, List[Pos]], bool, int]:
    """
    Returns (paths, budget_exhausted, nodes_expanded).
    """
    # --- Root: plan each agent with exact-cell t=0 occupancy constraints ---
    root_paths: Dict[int, List[Pos]] = {}

    for plane in agents:
        aid        = plane["id"]
        t0_blocked = _occupancy_constraints_at_t0(agents, aid)

        path = _spacetime_astar(
            start=plane["pos"],
            goal=goal,
            grid_size=grid_size,
            constraints=t0_blocked,
            edge_constraints=set(),
        )
        root_paths[aid] = path if path is not None else [plane["pos"]]

    root    = CTNode([], root_paths)
    heap    = []
    counter = 0
    heapq.heappush(heap, (root.cost, counter, root))

    nodes_expanded = 0

    while heap and nodes_expanded < max_nodes:
        _, _, node = heapq.heappop(heap)
        nodes_expanded += 1

        conflict = _find_first_conflict(node.paths)
        if conflict is None:
            return node.paths, False, nodes_expanded

        a, b, loc, t, ctype = conflict

        for constrained_agent in (a, b):
            # FIX: proper vertex vs edge constraints, no overwrite bug
            if ctype == "vertex":
                new_constraints = [(constrained_agent, loc, t)]
            else:
                # Edge conflict: block the swap move for each agent
                if constrained_agent == a:
                    new_constraints = [(constrained_agent, (loc[0], loc[1]), t)]
                else:
                    new_constraints = [(constrained_agent, (loc[1], loc[0]), t)]

            child_constraints = node.constraints + new_constraints
            vertex_c, edge_c  = _build_constraint_sets(child_constraints, constrained_agent)

            # FIX: do NOT re-add t=0 occupancy constraints inside the CT loop.
            # They fight CBS's own constraints and inflate the search tree.
            # The root A* already handled initial separation.

            new_path = _spacetime_astar(
                start=next(p["pos"] for p in agents if p["id"] == constrained_agent),
                goal=goal,
                grid_size=grid_size,
                constraints=vertex_c,
                edge_constraints=edge_c,
            )

            if new_path is None:
                continue

            child_paths = dict(node.paths)
            child_paths[constrained_agent] = new_path

            child   = CTNode(child_constraints, child_paths)
            counter += 1
            heapq.heappush(heap, (child.cost, counter, child))

    # Budget exhausted
    if heap:
        _, _, best_node = min(heap, key=lambda x: x[0])
        return best_node.paths, True, nodes_expanded

    return root_paths, True, nodes_expanded


# ---------------------------------------------------------------------------
# Public planner interface
# ---------------------------------------------------------------------------

def cbs_planner(
    grid:          "np.ndarray",
    active_planes: List[Dict],
    runway:        Pos,
) -> Dict[int, Pos]:
    global _step, _last_actions, _last_positions, _last_conflicts
    _step += 1

    if not active_planes:
        _last_actions   = {}
        _last_positions = {}
        _last_conflicts = []
        return {}

    grid_size   = grid.shape[0]
    id_to_plane = {p["id"]: p for p in active_planes}

    # --- TRACE 1: did last step's imminent conflicts carry over? ---
    if _last_conflicts:
        for a, b, loc, t, ctype, dist in _last_conflicts:
            if t > 1:
                continue
            pa = id_to_plane.get(a)
            pb = id_to_plane.get(b)
            if pa is None or pb is None:
                continue
            curr_dist = math.sqrt(
                (pa["pos"][0] - pb["pos"][0])**2 +
                (pa["pos"][1] - pb["pos"][1])**2
            )
            action_a = _last_actions.get(a, "none")
            action_b = _last_actions.get(b, "none")
            prev_a   = _last_positions.get(a, "?")
            prev_b   = _last_positions.get(b, "?")
            msg = (
                f"[Step {_step}] 🔍 TRACE conflict {a}&{b} from last step:\n"
                f"           last step: {a} was at {prev_a} → issued move {action_a}\n"
                f"           last step: {b} was at {prev_b} → issued move {action_b}\n"
                f"           now: {a} at {pa['pos']}, {b} at {pb['pos']}, dist={curr_dist:.2f}"
            )
            print(msg); logging.info(msg)
            if curr_dist < WARNING_RADIUS:
                m2 = "           💥 CONFIRMED: conflict carried over into this step!"
                print(m2); logging.info(m2)

    # --- TRACE 2: pre-existing violations before CBS runs ---
    pre_existing = []
    pids = [p["id"] for p in active_planes]
    for i in range(len(pids)):
        for j in range(i+1, len(pids)):
            a, b = pids[i], pids[j]
            pa, pb = id_to_plane[a], id_to_plane[b]
            d = math.sqrt(
                (pa["pos"][0]-pb["pos"][0])**2 +
                (pa["pos"][1]-pb["pos"][1])**2
            )
            if d < WARNING_RADIUS:
                pre_existing.append((a, b, pa["pos"], pb["pos"], d))

    if pre_existing:
        msg = (f"[Step {_step}] 🚨 PRE-EXISTING violations BEFORE CBS runs "
               f"({len(pre_existing)} pairs) — CBS CANNOT fix these:")
        print(msg); logging.info(msg)
        for a, b, pos_a, pos_b, d in pre_existing:
            m2 = f"           agents {a}&{b} | {pos_a} vs {pos_b} | dist={d:.2f}"
            print(m2); logging.info(m2)

    # --- Run CBS ---
    paths, budget_exhausted, nodes_expanded = _cbs(active_planes, runway, grid_size)

    if budget_exhausted:
        msg = (f"[Step {_step}] ⚠️  CBS BUDGET EXHAUSTED after {nodes_expanded} nodes "
               f"({len(active_planes)} agents)")
        print(msg); logging.info(msg)
    else:
        msg = (f"[Step {_step}] ✅ CBS solved cleanly in {nodes_expanded} nodes "
               f"({len(active_planes)} agents)")
        print(msg); logging.info(msg)

    # --- TRACE 3: imminent (t≤1) conflicts ---
    remaining = _find_all_conflicts(paths)
    imminent  = [(a,b,loc,t,ct,d) for a,b,loc,t,ct,d in remaining if t <= 1]
    if imminent:
        msg = (f"[Step {_step}] ❌ IMMINENT conflicts (t≤1) in CBS solution "
               f"— these will cause collisions NEXT step:")
        print(msg); logging.info(msg)
        for a, b, loc, t, ctype, dist in imminent:
            m2 = f"           agents {a}&{b} | t={t} | dist={dist:.2f} | loc={loc}"
            print(m2); logging.info(m2)
    else:
        msg = f"[Step {_step}] ✅ No imminent (t≤1) conflicts"
        print(msg); logging.info(msg)

    # --- Extract actions ---
    actions: Dict[int, Pos] = {}
    for plane in active_planes:
        pid      = plane["id"]
        path     = paths.get(pid, [])
        if len(path) < 2:
            continue
        next_pos = path[1]
        if next_pos == plane["pos"]:
            continue
        actions[pid] = next_pos

    # --- TRACE 4: duplicate moves = guaranteed collision next step ---
    next_pos_to_agents: Dict[Pos, List[int]] = {}
    for pid, npos in actions.items():
        next_pos_to_agents.setdefault(npos, []).append(pid)

    for npos, agent_list in next_pos_to_agents.items():
        if len(agent_list) > 1:
            msg = (f"[Step {_step}] 💣 DUPLICATE MOVE: agents {agent_list} "
                   f"all issued move to {npos} — collision guaranteed next step!")
            print(msg); logging.info(msg)

    # --- Save state for next step's trace ---
    _last_actions   = dict(actions)
    _last_positions = {p["id"]: p["pos"] for p in active_planes}
    _last_conflicts = remaining

    return actions