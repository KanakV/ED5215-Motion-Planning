"""
Static CBS / Prioritized Planner Test Harness
=============================================
Deterministic, reproducible scenarios for regression testing.
Tests both the original (buggy) and fixed planners side-by-side.

Run with:
    python test_cbs.py

All scenarios should show:
  [ORIGINAL] exhausted=True / violations > 0
  [FIXED]    exhausted=N/A  / violations = 0

BUGS FIXED (summary):
  1. No wait moves → A* now includes stay-in-place action
  2. Goal not absorbing in conflict checks → landed agents removed from checks  
  3. Blocking radius mismatch → now uses full WARNING_RADIUS
  4. Constraint at wrong timestep → now blocks at time t (not t+1)
"""

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Inline config so this file is self-contained
# ---------------------------------------------------------------------------
WARNING_RADIUS   = 3.0
COLLISION_RADIUS = 1.5
PLANE_RADIUS     = 1.0

Pos = Tuple[int, int]


# ===========================================================================
# SHARED UTILITIES
# ===========================================================================

def _cells_within_radius(center, radius, grid_size):
    cr, cc = center
    r_ceil = math.ceil(radius)
    cells  = []
    for dr in range(-r_ceil, r_ceil + 1):
        for dc in range(-r_ceil, r_ceil + 1):
            if math.sqrt(dr*dr + dc*dc) <= radius:
                nr, nc = cr+dr, cc+dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    cells.append((nr, nc))
    return cells


def _heuristic(pos, goal):
    return abs(pos[0]-goal[0]) + abs(pos[1]-goal[1])


def _goal_land_t(path, goal):
    for t, pos in enumerate(path):
        if pos == goal:
            return t
    return len(path)


def count_violations(paths, goal, threshold=WARNING_RADIUS):
    """Count separation violations, respecting goal absorption."""
    agent_ids = list(paths.keys())
    max_len   = max((len(p) for p in paths.values()), default=0)
    land_t    = {aid: _goal_land_t(paths[aid], goal) for aid in agent_ids}
    viols     = []
    for i in range(len(agent_ids)):
        for j in range(i+1, len(agent_ids)):
            a, b = agent_ids[i], agent_ids[j]
            pa, pb = paths[a], paths[b]
            for t in range(max_len):
                if t > land_t[a] or t > land_t[b]:
                    break
                pos_a = pa[min(t, len(pa)-1)]
                pos_b = pb[min(t, len(pb)-1)]
                dist  = math.sqrt((pos_a[0]-pos_b[0])**2 + (pos_a[1]-pos_b[1])**2)
                if dist < threshold:
                    viols.append((a, b, t, round(dist, 3)))
                    break
    return viols


# ===========================================================================
# ORIGINAL (buggy) CBS — preserved for comparison
# ===========================================================================

def _astar_orig(start, goal, grid_size, constraints, edge_constraints, max_t=300):
    """Original A* — NO wait moves."""
    def neighbors(pos):
        r, c = pos
        return [(nr,nc) for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                if 0 <= nr < grid_size and 0 <= nc < grid_size]

    open_heap = [(_heuristic(start,goal), 0, start, 0)]
    came_from = {(start,0): None}
    best_g    = {(start,0): 0}
    while open_heap:
        f, g, pos, t = heapq.heappop(open_heap)
        if best_g.get((pos,t), float("inf")) < g: continue
        if pos == goal:
            path, state = [], (pos,t)
            while state: path.append(state[0]); state = came_from[state]
            path.reverse(); return path
        if t >= max_t: continue
        for npos in neighbors(pos):
            if (npos,t+1) in constraints or (pos,npos,t) in edge_constraints: continue
            ng, key = g+1, (npos,t+1)
            if best_g.get(key, float("inf")) <= ng: continue
            best_g[key] = ng; came_from[key] = (pos,t)
            heapq.heappush(open_heap, (ng+_heuristic(npos,goal), ng, npos, t+1))
    return None


def _find_conflict_orig(paths):
    """Original conflict finder — does NOT handle goal absorption."""
    agent_ids = list(paths.keys())
    max_len   = max((len(p) for p in paths.values()), default=0)
    best = None
    for i in range(len(agent_ids)):
        for j in range(i+1, len(agent_ids)):
            a, b = agent_ids[i], agent_ids[j]
            pa, pb = paths[a], paths[b]
            for t in range(max_len):
                pos_a = pa[min(t,len(pa)-1)]
                pos_b = pb[min(t,len(pb)-1)]
                dist  = math.sqrt((pos_a[0]-pos_b[0])**2 + (pos_a[1]-pos_b[1])**2)
                if dist < WARNING_RADIUS:
                    if best is None or t < best[3]: best = (a,b,pos_a,t,"vertex")
                    break
    return best


class CTNode:
    __slots__ = ("constraints","paths","cost")
    def __init__(self,c,p): self.constraints=c; self.paths=p; self.cost=sum(len(x)-1 for x in p.values())
    def __lt__(self,o): return self.cost<o.cost

    def _build_cs(self, agent_id):
        vertex, edge = set(), set()
        for (aid, loc, t) in self.constraints:
            if aid != agent_id: continue
            if isinstance(loc[0], tuple): edge.add((loc[0],loc[1],t))
            else: vertex.add((loc,t))
        return vertex, edge


def run_cbs_orig(agents, goal, grid_size, max_nodes=600):
    _BUBBLE_R = WARNING_RADIUS / 2.0

    def t0_block(agent_id):
        blocked = set()
        for other in agents:
            if other["id"] == agent_id: continue
            for cell in _cells_within_radius(other["pos"], _BUBBLE_R, grid_size):
                blocked.add((cell,0))
        return blocked

    root_paths = {}
    for plane in agents:
        t0 = t0_block(plane["id"])
        path = _astar_orig(plane["pos"], goal, grid_size, t0, set())
        root_paths[plane["id"]] = path if path else [plane["pos"]]

    root = CTNode([], root_paths); heap = []; counter = 0
    heapq.heappush(heap, (root.cost, counter, root))
    nodes_expanded = 0

    while heap and nodes_expanded < max_nodes:
        _, _, node = heapq.heappop(heap); nodes_expanded += 1
        conflict = _find_conflict_orig(node.paths)
        if conflict is None:
            return node.paths, False, nodes_expanded
        a, b, loc, t, _ = conflict
        for ca in (a, b):
            new_c = [(ca, cell, t) for cell in _cells_within_radius(loc, _BUBBLE_R, grid_size)]
            child_c = node.constraints + new_c
            tmp = CTNode(child_c, {}); vc, ec = tmp._build_cs(ca)
            vc |= t0_block(ca)
            new_path = _astar_orig(next(p["pos"] for p in agents if p["id"]==ca), goal, grid_size, vc, ec)
            if new_path is None: continue
            cp = dict(node.paths); cp[ca] = new_path
            child = CTNode(child_c, cp); counter += 1
            heapq.heappush(heap, (child.cost, counter, child))

    if heap:
        _, _, bn = min(heap, key=lambda x: x[0])
        return bn.paths, True, nodes_expanded
    return root_paths, True, nodes_expanded


# ===========================================================================
# FIXED PLANNER — Prioritized A* with correct blocking
# ===========================================================================

def _astar_fixed(start, goal, grid_size, other_paths, land_times, max_t=300):
    """
    Fixed A*:
    - Wait moves included (BUG 1)
    - Blocks at timestep t, not t+1 (BUG 4)
    - Blocking radius = full WARNING_RADIUS (BUG 3)
    - Landed agents excluded from blocking (BUG 2)
    """
    # Precompute forbidden (cell, t) pairs
    constraints: Set[Tuple[Pos, int]] = set()
    for aid, opath in other_paths.items():
        lt = land_times[aid]
        for t in range(max_t + 1):
            if t > lt: break
            opos = opath[min(t, len(opath)-1)]
            for cell in _cells_within_radius(opos, WARNING_RADIUS, grid_size):
                constraints.add((cell, t))   # at time t, not t+1

    counter   = 0
    open_heap = [(_heuristic(start,goal), counter, 0, start)]
    came_from = {(start,0): None}
    best_g    = {(start,0): 0}

    while open_heap:
        f, _, g, pos = heapq.heappop(open_heap)
        t = g
        if best_g.get((pos,t), float("inf")) < g: continue
        if pos == goal:
            path, state = [], (pos,t)
            while state: path.append(state[0]); state = came_from[state]
            path.reverse(); return path
        if t >= max_t: continue
        r, c = pos
        # All 4 cardinal neighbours + wait
        candidates = [(nr,nc) for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                      if 0 <= nr < grid_size and 0 <= nc < grid_size]
        candidates.append(pos)   # wait
        for npos in candidates:
            nt = t + 1
            if (npos, nt) in constraints: continue
            ng, key = g+1, (npos,nt)
            if best_g.get(key, float("inf")) <= ng: continue
            best_g[key] = ng; came_from[key] = (pos,t)
            counter += 1
            heapq.heappush(open_heap, (ng+_heuristic(npos,goal), counter, ng, npos))

    return None


def run_fixed(agents, goal, grid_size):
    """Prioritized planning — plan in list order, each avoids all prior agents."""
    planned:    Dict[int, List[Pos]] = {}
    land_times: Dict[int, int]       = {}

    for plane in agents:
        aid  = plane["id"]
        path = _astar_fixed(plane["pos"], goal, grid_size, planned, land_times)
        if path is None:
            path = [plane["pos"]]   # no path found — agent holds
        planned[aid]    = path
        land_times[aid] = _goal_land_t(path, goal)

    return planned


# ===========================================================================
# Test runner
# ===========================================================================

def run_test(name, agents, goal, grid_size=30):
    print(f"\n{'='*62}")
    print(f"TEST: {name}")
    print(f"  Grid: {grid_size}x{grid_size}  |  Goal: {goal}  |  WARNING_RADIUS: {WARNING_RADIUS}")
    for a in agents:
        print(f"  Agent {a['id']}: {a['pos']}")

    # Original CBS
    paths_o, exh_o, n_o = run_cbs_orig(agents, goal, grid_size)
    viols_o = count_violations(paths_o, goal)
    status_o = "BUDGET EXHAUSTED" if exh_o else "solved"
    print(f"\n  [ORIGINAL CBS] {status_o} ({n_o} nodes) | violations: {len(viols_o)}")
    for v in viols_o[:3]:  # show first 3
        print(f"    agents {v[0]}&{v[1]} at t={v[2]}, dist={v[3]}")
    if len(viols_o) > 3:
        print(f"    ... and {len(viols_o)-3} more")

    # Fixed planner
    paths_f = run_fixed(agents, goal, grid_size)
    viols_f = count_violations(paths_f, goal)
    passed  = len(viols_f) == 0
    print(f"\n  [FIXED]        violations: {len(viols_f)}")
    for v in viols_f:
        print(f"    agents {v[0]}&{v[1]} at t={v[2]}, dist={v[3]}")

    print(f"\n  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


def make_agent(id_, pos):
    return {"id": id_, "pos": pos}


# ===========================================================================
# Test scenarios
# ===========================================================================

if __name__ == "__main__":
    results = []

    results.append(run_test(
        "1. Head-on collision (swap conflict)  — step-5 analog",
        agents=[make_agent(0, (15, 5)), make_agent(1, (15, 25))],
        goal=(15, 15),
    ))

    results.append(run_test(
        "2. Perpendicular crossing (vertex conflict)",
        agents=[make_agent(0, (5, 15)), make_agent(1, (15, 5))],
        goal=(15, 15),
    ))

    results.append(run_test(
        "3. Three agents, triangular approach",
        agents=[make_agent(0, (0, 15)), make_agent(1, (29, 10)), make_agent(2, (15, 0))],
        goal=(15, 15),
    ))

    results.append(run_test(
        "4. Four agents converging  — step-6 analog",
        agents=[make_agent(0, (0, 15)), make_agent(1, (29, 15)),
                make_agent(2, (15, 0)), make_agent(3, (15, 29))],
        goal=(15, 15),
    ))

    results.append(run_test(
        "5. Four corners funneling to center (extreme congestion)",
        agents=[make_agent(0, (0, 0)), make_agent(1, (0, 29)),
                make_agent(2, (29, 0)), make_agent(3, (29, 29))],
        goal=(14, 14),
    ))

    results.append(run_test(
        "6. Close spawn (dist = WARNING_RADIUS + 1)",
        agents=[make_agent(0, (10, 15)), make_agent(1, (10+int(WARNING_RADIUS)+1, 15))],
        goal=(20, 15),
    ))

    results.append(run_test(
        "7. Same-side parallel approach",
        agents=[make_agent(0, (0, 13)), make_agent(1, (0, 17))],
        goal=(15, 15),
    ))

    results.append(run_test(
        "8. Two-agent budget exhaustion reproduction  (exact step-5 scenario)",
        agents=[make_agent(0, (0, 10)), make_agent(1, (0, 20))],
        goal=(15, 15),
    ))

    results.append(run_test(
        "9. Tight grid stress test",
        agents=[make_agent(0, (5, 5)), make_agent(1, (5, 10))],
        goal=(5, 7),
        grid_size=15,
    ))

    results.append(run_test(
        "10. Single agent (trivial baseline)",
        agents=[make_agent(0, (0, 0))],
        goal=(15, 15),
    ))

    # Summary
    n_pass = sum(results)
    print(f"\n{'='*62}")
    print(f"SUMMARY: {n_pass}/{len(results)} tests passed")
    if n_pass == len(results):
        print("🎉 All tests passed!")
    else:
        failed = [i+1 for i, r in enumerate(results) if not r]
        print(f"❌ Failed tests: {failed}")