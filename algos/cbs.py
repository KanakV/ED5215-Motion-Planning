"""
Conflict-Based Search (CBS) Planner
=====================================
Genuine CBS implementation using a Constraint Tree (CT).

HOW CBS WORKS
-------------
CBS is a two-level search algorithm for multi-agent path finding (MAPF):

  HIGH LEVEL — Constraint Tree (CT)
    Each CT node stores:
      • constraints  : list of (agent_id, cell, timestep) triples forbidding
                       specific agents from specific cells at specific times
      • paths        : one complete path per agent, planned under those constraints
      • cost         : sum of individual path lengths (SIC — sum of individual costs)

    The root has no constraints; each agent's path is its unconstrained optimum.
    CBS pops the lowest-cost CT node from a min-heap, checks all paths for
    inter-agent conflicts, and if one is found it branches:

      Conflict (agent_a, agent_b) at time t:
        Child A → constrain agent_a away from agent_b's position at t, replan agent_a
        Child B → constrain agent_b away from agent_a's position at t, replan agent_b

    CBS terminates when a CT node has no conflicts (optimal solution found) or
    the node budget is exhausted (fallback to best found so far).

  LOW LEVEL — Space-Time A*
    Plans a single agent's shortest path from its start to the goal, honouring
    its accumulated set of (cell, timestep) vertex constraints. Each constraint
    (cell, t) blocks the agent from occupying `cell` at timestep `t`.

    Delegated to the unified spacetime_astar() function in algos/spacetime_astar.py,
    called with grid_size= and constraints=.

BUGS FIXED VS ORIGINAL
-----------------------
1. CONSTRAINT RADIUS was WARNING_RADIUS/2 — too small.
   Conflict detection fires at dist < WARNING_RADIUS, but constraints only
   blocked cells within WARNING_RADIUS/2. So replanned paths still conflicted.
   Fix: constraint radius = WARNING_RADIUS.

2. CONSTRAINT DIRECTION was wrong — constrained the agent from its OWN position
   instead of from the OTHER agent's position.
   Fix: block constrained_agent from cells near other_agent's conflict position.

3. GOAL NOT ABSORBING in conflict detection — paths were checked past landing,
   treating a landed plane as permanently occupying the runway. Every subsequent
   agent's approach conflicted with the ghost, making the problem unsolvable.
   Fix: stop checking agent at timesteps beyond its landing time.

NOTE: No wait moves — agents must move to a cardinal neighbour every step.
      Some configurations (e.g. all four corners converging simultaneously)
      may be unsolvable without waits; CBS will exhaust its budget and return
      the best paths found.
"""

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

from config.config import PLANE_RADIUS, WARNING_RADIUS, MAX_NODES
from algos.spacetime_astar import spacetime_astar
from algos.algo_helpers import manhattan

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Pos        = Tuple[int, int]
Constraint = Tuple[int, Pos, int]   # (agent_id, cell, timestep)

_step            = 0
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


def _goal_land_t(path: List[Pos], goal: Pos) -> int:
    """First timestep at which agent occupies the goal cell."""
    for t, pos in enumerate(path):
        if pos == goal:
            return t
    return len(path)


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

def _extract_agent_constraints(
    all_constraints: List[Constraint],
    agent_id:        int,
) -> Set[Tuple[Pos, int]]:
    """Pull out the (cell, timestep) constraint set for one agent."""
    return {(cell, t) for (aid, cell, t) in all_constraints if aid == agent_id}


def _make_constraints(
    constrained_agent: int,
    other_pos:         Pos,
    conflict_t:        int,
    grid_size:         int,
) -> List[Constraint]:
    """
    Constraints that keep `constrained_agent` >= WARNING_RADIUS from `other_pos`
    at `conflict_t`. Blocks every cell within WARNING_RADIUS of other_pos.
    """
    return [
        (constrained_agent, cell, conflict_t)
        for cell in _cells_within_radius(other_pos, WARNING_RADIUS, grid_size)
    ]


# ---------------------------------------------------------------------------
# High level: conflict detection
# ---------------------------------------------------------------------------

def _find_first_conflict(
    paths: Dict[int, List[Pos]],
    goal:  Pos,
) -> Optional[Tuple[int, int, Pos, Pos, int]]:
    """
    Find the earliest-timestep conflict across all agent pairs.
    Returns (agent_a, agent_b, pos_a, pos_b, t) or None if conflict-free.

    Goal absorption: once an agent lands it leaves the airspace — stop
    checking it for conflicts beyond its landing timestep.
    """
    agent_ids = list(paths.keys())
    max_len   = max((len(p) for p in paths.values()), default=0)
    land_t    = {aid: _goal_land_t(paths[aid], goal) for aid in agent_ids}
    earliest  = None

    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            a, b   = agent_ids[i], agent_ids[j]
            pa, pb = paths[a], paths[b]

            for t in range(max_len):
                if t > land_t[a] or t > land_t[b]:
                    break

                pos_a = pa[min(t, len(pa) - 1)]
                pos_b = pb[min(t, len(pb) - 1)]
                dist  = math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)

                if dist < WARNING_RADIUS:
                    if earliest is None or t < earliest[4]:
                        earliest = (a, b, pos_a, pos_b, t)
                    break

    return earliest


def _find_all_conflicts(
    paths: Dict[int, List[Pos]],
    goal:  Pos,
) -> List[Tuple]:
    """Find ALL conflicts — used for diagnostics only."""
    agent_ids = list(paths.keys())
    max_len   = max((len(p) for p in paths.values()), default=0)
    land_t    = {aid: _goal_land_t(paths[aid], goal) for aid in agent_ids}
    conflicts = []
    seen      = set()

    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            a, b   = agent_ids[i], agent_ids[j]
            pa, pb = paths[a], paths[b]

            for t in range(max_len):
                if t > land_t[a] or t > land_t[b]:
                    break

                pos_a = pa[min(t, len(pa) - 1)]
                pos_b = pb[min(t, len(pb) - 1)]
                dist  = math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
                pair  = (min(a, b), max(a, b))

                if dist < WARNING_RADIUS and pair not in seen:
                    seen.add(pair)
                    conflicts.append((a, b, pos_a, t, dist))
                    break

    return conflicts


# ---------------------------------------------------------------------------
# High level: CT Node
# ---------------------------------------------------------------------------

class CTNode:
    """
    A node in the CBS Constraint Tree.

    constraints : all (agent_id, cell, timestep) constraints accumulated to here
    paths       : complete path set for all agents under these constraints
    cost        : sum of individual path lengths (SIC)
    """
    __slots__ = ("constraints", "paths", "cost")

    def __init__(self, constraints: List[Constraint], paths: Dict[int, List[Pos]]):
        self.constraints = constraints
        self.paths       = paths
        self.cost        = sum(len(p) - 1 for p in paths.values())

    def __lt__(self, other: "CTNode"):
        return self.cost < other.cost


# ---------------------------------------------------------------------------
# High level: CBS
# ---------------------------------------------------------------------------

def _cbs(
    agents:    List[Dict],
    goal:      Pos,
    grid_size: int,
    max_nodes: int = MAX_NODES,
) -> Tuple[Dict[int, List[Pos]], bool, int]:
    """
    Run CBS. Returns (paths, budget_exhausted, nodes_expanded).

    Low-level planner: spacetime_astar(..., grid_size=grid_size, constraints=...).
    """

    # ---- Root node: unconstrained individual plans ----
    root_paths: Dict[int, List[Pos]] = {}
    for plane in agents:
        path = spacetime_astar(
            start=plane["pos"],
            goal=goal,
            grid_size=grid_size,
            constraints=set(),
        )
        root_paths[plane["id"]] = path if path is not None else [plane["pos"]]

    root    = CTNode([], root_paths)
    heap    = []
    counter = 0
    heapq.heappush(heap, (root.cost, counter, root))
    nodes_expanded = 0

    while heap and nodes_expanded < max_nodes:
        _, _, node = heapq.heappop(heap)
        nodes_expanded += 1

        conflict = _find_first_conflict(node.paths, goal)
        if conflict is None:
            return node.paths, False, nodes_expanded   # optimal solution

        agent_a, agent_b, pos_a, pos_b, t = conflict

        # Branch: each child constrains one agent away from the other's position
        for constrained_agent, other_pos in [(agent_a, pos_b), (agent_b, pos_a)]:

            new_constraints   = _make_constraints(constrained_agent, other_pos, t, grid_size)
            child_constraints = node.constraints + new_constraints

            agent_cs  = _extract_agent_constraints(child_constraints, constrained_agent)
            start_pos = next(p["pos"] for p in agents if p["id"] == constrained_agent)

            new_path = spacetime_astar(
                start=start_pos,
                goal=goal,
                grid_size=grid_size,
                constraints=agent_cs,
            )
            if new_path is None:
                continue   # no valid path under these constraints — prune branch

            child_paths = dict(node.paths)
            child_paths[constrained_agent] = new_path

            child   = CTNode(child_constraints, child_paths)
            counter += 1
            heapq.heappush(heap, (child.cost, counter, child))

    # ---- Budget exhausted ----
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

    # ------------------------------------------------------------------
    # TRACE 1: Did last step's conflicts carry over?
    # ------------------------------------------------------------------
    if _last_conflicts:
        for a, b, loc, t, dist in _last_conflicts:
            if t != 0:
                continue
            pa = id_to_plane.get(a)
            pb = id_to_plane.get(b)
            if pa is None or pb is None:
                continue
            curr_dist = math.sqrt(
                (pa["pos"][0] - pb["pos"][0])**2 +
                (pa["pos"][1] - pb["pos"][1])**2
            )
            print(
                f"[Step {_step}] 🔍 TRACE conflict {a}&{b} from last step: "
                f"{a}@{_last_positions.get(a,'?')}→{_last_actions.get(a,'none')}, "
                f"{b}@{_last_positions.get(b,'?')}→{_last_actions.get(b,'none')} | "
                f"now dist={curr_dist:.2f}"
            )
            if curr_dist < WARNING_RADIUS:
                print(f"           💥 CONFIRMED: conflict carried over!")

    # ------------------------------------------------------------------
    # TRACE 2: Pre-existing violations
    # ------------------------------------------------------------------
    pids = [p["id"] for p in active_planes]
    pre  = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = pids[i], pids[j]
            d = math.sqrt(
                (id_to_plane[a]["pos"][0] - id_to_plane[b]["pos"][0])**2 +
                (id_to_plane[a]["pos"][1] - id_to_plane[b]["pos"][1])**2
            )
            if d < WARNING_RADIUS:
                pre.append((a, b, id_to_plane[a]["pos"], id_to_plane[b]["pos"], d))

    if pre:
        print(f"[Step {_step}] 🚨 PRE-EXISTING violations ({len(pre)} pairs):")
        for a, b, pa, pb, d in pre:
            print(f"           agents {a}&{b} | {pa} vs {pb} | dist={d:.2f}")

    # ------------------------------------------------------------------
    # Run CBS
    # ------------------------------------------------------------------
    paths, budget_exhausted, nodes_expanded = _cbs(active_planes, runway, grid_size)

    if budget_exhausted:
        print(f"[Step {_step}] ⚠️  CBS budget exhausted ({nodes_expanded} nodes, "
              f"{len(active_planes)} agents)")

    # ------------------------------------------------------------------
    # TRACE 3: Solution quality
    # ------------------------------------------------------------------
    remaining = _find_all_conflicts(paths, runway)
    imminent  = [(a, b, loc, t, d) for a, b, loc, t, d in remaining if t <= 1]
    if imminent:
        print(f"[Step {_step}] ❌ IMMINENT conflicts (t≤1) in CBS solution:")
        for a, b, loc, t, dist in imminent:
            print(f"           agents {a}&{b} | t={t} | dist={dist:.2f} | loc={loc}")

    # ------------------------------------------------------------------
    # Extract next-step actions
    # ------------------------------------------------------------------
    actions: Dict[int, Pos] = {}
    for plane in active_planes:
        pid  = plane["id"]
        path = paths.get(pid, [])
        if len(path) < 2:
            continue
        next_pos = path[1]
        if next_pos == plane["pos"]:
            continue
        actions[pid] = next_pos

    # ------------------------------------------------------------------
    # TRACE 4: Duplicate move check
    # ------------------------------------------------------------------
    nxt: Dict[Pos, List[int]] = {}
    for pid, np_ in actions.items():
        nxt.setdefault(np_, []).append(pid)
    for np_, agent_list in nxt.items():
        if len(agent_list) > 1:
            print(f"[Step {_step}] 💣 DUPLICATE MOVE: agents {agent_list} → {np_}")

    # ------------------------------------------------------------------
    # Save state for next step diagnostics
    # ------------------------------------------------------------------
    _last_actions   = dict(actions)
    _last_positions = {p["id"]: p["pos"] for p in active_planes}
    _last_conflicts = remaining

    return actions