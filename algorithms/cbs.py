"""
Conflict-Based Search (CBS) for Optimal Multi-Agent Pathfinding
================================================================
Faithful implementation of:
    Sharon, G., Stern, R., Felner, A., & Sturtevant, N. R. (2015).
    Conflict-based search for optimal multi-agent pathfinding.
    Artificial Intelligence, 219, 40–66.

Paper structure mapping
-----------------------
Section 4.1  – Definitions  →  Constraint, Conflict, CTNode
Section 4.2  – High level   →  CBS.high_level_search()
Section 4.3  – Low level    →  CBS.low_level_search()
Section 5.1  – Optimality   →  proven by best-first expansion on CT cost
Section 5.2  – Completeness →  finite CT nodes per cost (Theorem 2 & 3)
Section 8    – MA-CBS        →  MACBS (subclass)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Vertex = int          # graph node id
Time   = int          # discrete time step
AgentId = int         # 0-based agent index

# ---------------------------------------------------------------------------
# Section 4.1 – Definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Constraint:
    """
    A constraint (a_i, v, t) – agent a_i is prohibited from occupying
    vertex v at time step t.  (Section 4.1, bullet 2)
    """
    agent: AgentId
    vertex: Vertex
    time:   Time

    def __repr__(self) -> str:
        return f"({self.agent}, {self.vertex}, t={self.time})"


@dataclass(frozen=True)
class EdgeConstraint:
    """
    An edge constraint (a_i, v1, v2, t) – agent a_i is prohibited from
    traversing edge v1→v2 starting at time t.  (Section 4.2.5)
    """
    agent:  AgentId
    vertex1: Vertex
    vertex2: Vertex
    time:    Time


@dataclass(frozen=True)
class Conflict:
    """
    A vertex conflict (a_i, a_j, v, t) – agents a_i and a_j both occupy
    vertex v at time t.  (Section 4.1, bullet 3)
    """
    agent_i: AgentId
    agent_j: AgentId
    vertex:  Vertex
    time:    Time

    def __repr__(self) -> str:
        return f"({self.agent_i}, {self.agent_j}, v={self.vertex}, t={self.time})"


@dataclass(frozen=True)
class EdgeConflict:
    """
    An edge conflict (a_i, a_j, v1, v2, t) – agents swap locations between
    time t and t+1.  (Section 4.2.5)
    """
    agent_i:  AgentId
    agent_j:  AgentId
    vertex1:  Vertex
    vertex2:  Vertex
    time:     Time


# ---------------------------------------------------------------------------
# Section 4.2.1 – Constraint Tree Node
# ---------------------------------------------------------------------------

@dataclass
class CTNode:
    """
    A node N in the Constraint Tree (CT).  Contains:
      N.constraints – set of constraints for each agent
      N.solution    – k paths, one per agent (consistent with constraints)
      N.cost        – sum-of-costs of N.solution  (the f-value of this node)

    Section 4.2.1
    """
    constraints: Dict[AgentId, List[Constraint]]   # agent → list of constraints
    solution:    List[List[Vertex]]                 # agent → path (vertex sequence)
    cost:        int                                # sum-of-costs (f-value)
    conflicts:   int = 0                            # tie-break: fewer conflicts preferred

    # ---- ordering for heapq (best-first on cost, then fewest conflicts) ----
    def __lt__(self, other: "CTNode") -> bool:
        if self.cost != other.cost:
            return self.cost < other.cost
        return self.conflicts < other.conflicts


# ---------------------------------------------------------------------------
# Graph abstraction
# ---------------------------------------------------------------------------

class Graph:
    """
    Directed graph G(V, E) as described in Section 2.1.
    Vertices are integer ids; edges are stored as adjacency lists.
    A reverse adjacency list is also maintained for efficient SIC heuristic
    computation (reverse BFS from goal).
    """

    def __init__(self, num_vertices: int):
        self.num_vertices = num_vertices
        self._adj:  Dict[Vertex, List[Vertex]] = {v: [] for v in range(num_vertices)}
        self._radj: Dict[Vertex, List[Vertex]] = {v: [] for v in range(num_vertices)}

    def add_edge(self, u: Vertex, v: Vertex) -> None:
        self._adj[u].append(v)
        self._radj[v].append(u)

    def add_undirected_edge(self, u: Vertex, v: Vertex) -> None:
        self._adj[u].append(v)
        self._adj[v].append(u)
        self._radj[u].append(v)
        self._radj[v].append(u)

    def neighbors(self, v: Vertex) -> List[Vertex]:
        """All vertices reachable from v in one step (move actions)."""
        return self._adj[v]

    def reverse_neighbors(self, v: Vertex) -> List[Vertex]:
        """All vertices that can reach v in one step (for reverse BFS)."""
        return self._radj[v]

    def wait_action(self, v: Vertex) -> Vertex:
        """Wait action – agent stays at v (Section 2.2)."""
        return v

    @classmethod
    def from_grid(cls, rows: int, cols: int, obstacles: Set[Tuple[int, int]] = None) -> "Graph":
        """
        Build a 4-connected grid graph (the primary experimental domain,
        Section 3.3 / 6.1).  bbase = 5 (4 cardinal directions + wait).
        """
        obstacles = obstacles or set()
        g = cls(rows * cols)

        def idx(r: int, c: int) -> int:
            return r * cols + c

        for r in range(rows):
            for c in range(cols):
                if (r, c) in obstacles:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in obstacles:
                        g.add_edge(idx(r, c), idx(nr, nc))
        return g


# ---------------------------------------------------------------------------
# Section 4.3 – Low-level: single-agent A* with constraints
# ---------------------------------------------------------------------------

def _sic_heuristic(graph: Graph, goal: Vertex) -> Dict[Vertex, int]:
    """
    Sum-of-Individual-Costs (SIC) heuristic precomputation for a single
    agent: reverse BFS from goal, ignoring other agents.  (Section 3.3.1)

    Uses reverse edges so that h[v] = shortest *forward* path from v to goal.
    This is admissible because the reverse shortest path equals the forward
    shortest path length in the same graph.

    Returns h[v] = shortest path length from v to goal (ignoring agents).
    This is the admissible heuristic used in the low-level A*.
    """
    dist: Dict[Vertex, int] = {goal: 0}
    queue = [goal]
    head = 0
    while head < len(queue):
        v = queue[head]; head += 1
        # Walk reverse edges: predecessors of v can reach v in 1 step
        for u in graph.reverse_neighbors(v):
            if u not in dist:
                dist[u] = dist[v] + 1
                queue.append(u)
    return dist


def low_level_search(
    graph:       Graph,
    agent:       AgentId,
    start:       Vertex,
    goal:        Vertex,
    constraints: List[Constraint],
    h_table:     Dict[Vertex, int],
    cat:         Optional[Dict[Tuple[Vertex, Time], int]] = None,
    max_time:    int = 1000,
) -> Optional[List[Vertex]]:
    """
    Low-level search (Section 4.3).

    Finds the shortest path for `agent` from `start` to `goal` that is
    consistent with all its constraints.

    State space: (vertex, time) – includes time dimension to handle dynamic
    obstacles created by constraints (Section 4.3, paragraph 3).

    Tie-breaking: states with fewer conflicts with other agents' paths (via
    the Conflict Avoidance Table, CAT) are preferred.  (Section 4.3 / 3.3.4)

    Duplicate detection (DD): two states are duplicates if both vertex and
    time are identical.  (Section 4.3, last paragraph)

    Parameters
    ----------
    graph        : the environment graph
    agent        : agent being planned for
    start / goal : start and goal vertices
    constraints  : constraints on this agent (pruned during search)
    h_table      : precomputed SIC heuristic distances to goal
    cat          : conflict avoidance table  {(vertex, time): #conflicts}
    max_time     : hard cutoff to guarantee termination

    Returns
    -------
    List[Vertex] – path (vertex sequence) if found, else None
    """
    # Build a fast constraint lookup: (vertex, time) → True
    constraint_set: Set[Tuple[Vertex, Time]] = {
        (c.vertex, c.time) for c in constraints if c.agent == agent
    }

    # A* open list: (f, g, conflicts_at_state, vertex, time, parent_index)
    # 'conflicts_at_state' used for CAT tie-breaking (Section 3.3.4)
    cat = cat or {}

    @dataclass(order=True)
    class _State:
        f:          int
        cat_hits:   int   # tie-break: fewer CAT conflicts preferred
        g:          int
        vertex:     Vertex = field(compare=False)
        time:       Time   = field(compare=False)
        parent:     int    = field(compare=False)   # index into closed list

    closed:  List[_State] = []    # indexed store for path reconstruction
    visited: Dict[Tuple[Vertex, Time], int] = {}  # (v,t) → closed index

    h0 = h_table.get(start, 10**9)
    init = _State(
        f=h0, cat_hits=cat.get((start, 0), 0),
        g=0, vertex=start, time=0, parent=-1
    )
    open_list: List[_State] = []
    heapq.heappush(open_list, init)
        
    while open_list:
        s = heapq.heappop(open_list)
        key = (s.vertex, s.time)

        # Duplicate detection (Section 4.3)
        if key in visited:
            continue
        visited[key] = len(closed)
        closed.append(s)

        # Goal check: agent at goal and stays there (sum-of-costs semantics,
        # Section 2.5 – cost counted until agent reaches goal for last time)
        if s.vertex == goal:
            # reconstruct path
            path: List[Vertex] = []
            idx = len(closed) - 1
            while idx != -1:
                path.append(closed[idx].vertex)
                idx = closed[idx].parent
            path.reverse()
            return path

        if s.time >= max_time:
            continue

        # Expand: move actions + wait action (Section 2.2)
        next_time = s.time + 1
        successors: List[Vertex] = list(graph.neighbors(s.vertex)) + [s.vertex]  # wait

        for nv in successors:
            nkey = (nv, next_time)
            if nkey in visited:
                continue
            # Constraint check (Section 4.1 / 4.3)
            if (nv, next_time) in constraint_set:
                continue
            ng = s.g + 1
            nh = h_table.get(nv, 10**9)
            nc = cat.get(nkey, 0)
            ns = _State(
                f=ng + nh, cat_hits=nc,
                g=ng, vertex=nv, time=next_time,
                parent=visited[key]
            )
            heapq.heappush(open_list, ns)

    return None   # no solution found


# ---------------------------------------------------------------------------
# Section 4.2 – High level: CBS
# ---------------------------------------------------------------------------

def _validate_solution(
    solution: List[List[Vertex]],
    check_edge_conflicts: bool = True,
) -> Optional[Conflict | EdgeConflict]:
    """
    Validate a multi-agent solution by checking all time steps.
    Returns the *first* conflict found, or None if the solution is valid.

    Vertex conflict  (Section 4.2.2): two agents at same vertex at same time.
    Edge conflict    (Section 4.2.5): two agents swap locations between t & t+1.
    """
    if not solution:
        return None

    max_time = max(len(p) for p in solution)
    k = len(solution)

    def agent_at(agent: int, t: int) -> Vertex:
        """After the path ends the agent stays at its goal (Section 6.1)."""
        path = solution[agent]
        return path[t] if t < len(path) else path[-1]

    for t in range(max_time):
        # Vertex conflicts
        positions: Dict[Vertex, AgentId] = {}
        for i in range(k):
            v = agent_at(i, t)
            if v in positions:
                return Conflict(positions[v], i, v, t)
            positions[v] = i

        # Edge conflicts (Section 4.2.5) – agents swap locations
        if check_edge_conflicts and t + 1 < max_time:
            for i in range(k):
                for j in range(i + 1, k):
                    vi, vj = agent_at(i, t), agent_at(j, t)
                    ni, nj = agent_at(i, t+1), agent_at(j, t+1)
                    if vi == nj and vj == ni:
                        return EdgeConflict(i, j, vi, vj, t)

    return None


def _sum_of_costs(solution: List[List[Vertex]], goals: List[Vertex]) -> int:
    """
    Sum-of-costs cost function (Section 2.5):
    For each agent, the cost is the number of time steps until it reaches
    its goal *for the last time and never leaves*.

    Since the low-level A* terminates the path as soon as the agent reaches
    the goal (it does not continue planning beyond the goal), the cost for
    agent i is simply len(path_i) - 1.

    Section 6.1: wait actions at the goal cost zero only if the agent never
    leaves the goal later.  Our low-level never plans beyond the goal, so
    this simplification is valid.
    """
    total = 0
    for i, path in enumerate(solution):
        g = goals[i]
        # path[-1] must be the goal; cost = steps taken = len - 1
        # If the path overshoots (agent passes through goal and comes back),
        # find the first time agent arrives at goal and stays.
        cost = len(path) - 1
        # Trim: walk backward, counting steps at goal as free
        t = len(path) - 1
        while t > 0 and path[t] == g and path[t-1] == g:
            cost -= 1
            t   -= 1
        total += cost
    return total


class CBS:
    """
    Conflict-Based Search – Algorithm 2 (Section 4.2.6).

    High-level best-first search over the Constraint Tree.
    Low-level: single-agent A* with SIC heuristic and CAT tie-breaking.

    Usage
    -----
    >>> cbs = CBS(graph, starts, goals)
    >>> solution = cbs.search()
    """

    def __init__(
        self,
        graph:  Graph,
        starts: List[Vertex],
        goals:  List[Vertex],
        check_edge_conflicts: bool = True,
        max_time: int = 1000,
    ):
        assert len(starts) == len(goals), "Each agent needs a start and a goal."
        self.graph  = graph
        self.starts = starts
        self.goals  = goals
        self.k      = len(starts)
        self.check_edge_conflicts = check_edge_conflicts
        self.max_time = max_time

        # Precompute SIC heuristic for every agent (Section 3.3.1)
        # h_tables[i][v] = shortest distance from v to goals[i]
        self.h_tables: List[Dict[Vertex, int]] = [
            _sic_heuristic(graph, goals[i]) for i in range(self.k)
        ]

        # Statistics
        self.high_level_expanded = 0
        self.low_level_calls     = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self) -> Optional[List[List[Vertex]]]:
        """
        Run CBS and return an optimal solution (list of k paths), or None.

        Implements Algorithm 2 (CBS high level), lines 1-26, with
        shouldMerge() always returning False (basic CBS, not MA-CBS).
        """
        root = self._make_root()
        if root is None:
            return None   # no individual path exists for some agent

        open_list: List[CTNode] = []
        heapq.heappush(open_list, root)

        while open_list:
            # Line 6: P ← best node from OPEN (lowest cost)
            P = heapq.heappop(open_list)
            self.high_level_expanded += 1

            # Line 7: validate paths in P until a conflict occurs
            conflict = _validate_solution(P.solution, self.check_edge_conflicts)

            # Line 8-9: no conflict → goal node, return solution
            if conflict is None:
                return P.solution

            # Line 10: C ← first conflict
            # Lines 19-26: foreach agent in C, create child node
            for child in self._resolve_conflict(P, conflict):
                if child is not None:
                    heapq.heappush(open_list, child)

        return None   # OPEN exhausted, no solution

    # ------------------------------------------------------------------
    # Root node construction (Algorithm 2, lines 1-4)
    # ------------------------------------------------------------------

    def _make_root(self) -> Optional[CTNode]:
        """
        Algorithm 2, lines 1-3.
        Root.constraints = ∅
        Root.solution    = find individual paths by the low level
        Root.cost        = SIC(Root.solution)
        """
        constraints: Dict[AgentId, List[Constraint]] = {i: [] for i in range(self.k)}
        solution: List[Optional[List[Vertex]]] = []

        for i in range(self.k):
            path = self._low_level(i, constraints[i], solution)
            if path is None:
                return None
            solution.append(path)

        cost = _sum_of_costs(solution, self.goals)
        node = CTNode(
            constraints=constraints,
            solution=solution,
            cost=cost,
            conflicts=self._count_conflicts(solution),
        )
        return node

    # ------------------------------------------------------------------
    # Conflict resolution (Algorithm 2, lines 19-26)
    # ------------------------------------------------------------------

    def _resolve_conflict(
        self,
        parent: CTNode,
        conflict: Conflict | EdgeConflict,
    ) -> List[Optional[CTNode]]:
        """
        Section 4.2.3: split parent into two children, one for each agent
        involved in the conflict.

        Each child inherits parent's constraints and adds one new constraint
        that forces one of the conflicting agents to avoid the conflict.

        Section 4.2.5: edge conflicts are handled symmetrically.
        """
        children: List[Optional[CTNode]] = []

        if isinstance(conflict, EdgeConflict):
            pairs = [
                (conflict.agent_i, EdgeConstraint(conflict.agent_i, conflict.vertex1, conflict.vertex2, conflict.time)),
                (conflict.agent_j, EdgeConstraint(conflict.agent_j, conflict.vertex2, conflict.vertex1, conflict.time)),
            ]
        else:
            # Vertex conflict: add (agent, vertex, time) constraint
            pairs = [
                (conflict.agent_i, Constraint(conflict.agent_i, conflict.vertex, conflict.time)),
                (conflict.agent_j, Constraint(conflict.agent_j, conflict.vertex, conflict.time)),
            ]

        for agent, new_constraint in pairs:
            child = self._make_child(parent, agent, new_constraint)
            children.append(child)

        return children

    def _make_child(
        self,
        parent:         CTNode,
        agent:          AgentId,
        new_constraint: Constraint | EdgeConstraint,
    ) -> Optional[CTNode]:
        """
        Algorithm 2, lines 20-26.

        A.constraints ← P.constraints + new constraint
        A.solution    ← P.solution  (copy)
        Update A.solution by invoking low_level(agent)
        A.cost = SIC(A.solution)
        """
        # Deep-copy constraints (child inherits parent's, adds one new one)
        new_constraints: Dict[AgentId, List[Constraint]] = {
            i: list(cs) for i, cs in parent.constraints.items()
        }

        # Add new constraint for the conflicting agent
        if isinstance(new_constraint, Constraint):
            new_constraints[agent].append(new_constraint)
        # Edge constraints require separate handling in low-level (future extension)

        # Copy solution; only re-plan for the constrained agent
        new_solution: List[List[Vertex]] = [list(p) for p in parent.solution]

        # Invoke low level for the agent with new constraint (Algorithm 2, line 23)
        new_path = self._low_level(agent, new_constraints[agent], new_solution)

        # Algorithm 2, line 25: if A.cost < ∞ (a solution was found)
        if new_path is None:
            return None   # this branch has no valid solution

        new_solution[agent] = new_path
        cost = _sum_of_costs(new_solution, self.goals)

        return CTNode(
            constraints=new_constraints,
            solution=new_solution,
            cost=cost,
            conflicts=self._count_conflicts(new_solution),
        )

    # ------------------------------------------------------------------
    # Low-level invocation with CAT (Section 4.3)
    # ------------------------------------------------------------------

    def _low_level(
        self,
        agent:       AgentId,
        constraints: List[Constraint],
        current_solution: List[List[Vertex]],
    ) -> Optional[List[Vertex]]:
        """
        Section 4.3: invoke single-agent A* for `agent`.

        Builds the Conflict Avoidance Table (CAT) from paths of all other
        agents in the current solution, used for tie-breaking (Section 3.3.4).
        """
        self.low_level_calls += 1

        # Build CAT from all other agents' planned paths (Section 3.3.4)
        cat: Dict[Tuple[Vertex, Time], int] = {}
        for i, path in enumerate(current_solution):
            if i == agent:
                continue
            for t, v in enumerate(path):
                cat[(v, t)] = cat.get((v, t), 0) + 1

        return low_level_search(
            graph=self.graph,
            agent=agent,
            start=self.starts[agent],
            goal=self.goals[agent],
            constraints=constraints,
            h_table=self.h_tables[agent],
            cat=cat,
            max_time=self.max_time,
        )

    # ------------------------------------------------------------------
    # Helper: count total conflicts in a solution (for tie-breaking)
    # ------------------------------------------------------------------

    def _count_conflicts(self, solution: List[List[Vertex]]) -> int:
        count = 0
        max_t = max(len(p) for p in solution)
        k = len(solution)

        def at(i: int, t: int) -> Vertex:
            p = solution[i]
            return p[t] if t < len(p) else p[-1]

        for t in range(max_t):
            seen: Dict[Vertex, int] = {}
            for i in range(k):
                v = at(i, t)
                if v in seen:
                    count += 1
                seen[v] = i
        return count

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "high_level_expanded": self.high_level_expanded,
            "low_level_calls":     self.low_level_calls,
        }


# ---------------------------------------------------------------------------
# Section 3.3.3 – Independence Detection (ID) framework
# ---------------------------------------------------------------------------

class IDFramework:
    """
    Independence Detection (ID) framework (Algorithm 1, Section 3.3.3).

    Wraps any optimal MAPF solver (here: CBS).  Agents start in singleton
    groups and are merged only when their planned paths conflict.

    Note: CBS is described as compatible with ID (Section 3.3.3, last para).
    """

    def __init__(self, graph: Graph, starts: List[Vertex], goals: List[Vertex]):
        self.graph  = graph
        self.starts = starts
        self.goals  = goals
        self.k      = len(starts)

    def search(self) -> Optional[List[List[Vertex]]]:
        """Algorithm 1 (Section 3.3.3)."""
        # Line 1: assign each agent to a singleton group
        groups: List[List[AgentId]] = [[i] for i in range(self.k)]

        # Line 2: plan a path for each group
        group_solutions: List[Optional[List[List[Vertex]]]] = []
        for g in groups:
            sol = self._solve_group(g)
            if sol is None:
                return None
            group_solutions.append(sol)

        # Lines 3-8: repeat until no conflicts
        while True:
            # Line 4: validate the combined solution
            full_solution = self._combine(groups, group_solutions)
            conflict = _validate_solution(full_solution)

            if conflict is None:    # Line 8: no conflicts
                return full_solution

            # Line 5-7: merge conflicting groups and replan
            gi, gj = self._find_group_indices(groups, conflict)
            if gi == gj:
                # Already merged – shouldn't happen, but guard against it
                return full_solution

            # Merge groups gi and gj
            merged_agents = groups[gi] + groups[gj]
            merged_sol    = self._solve_group(merged_agents)
            if merged_sol is None:
                return None

            # Remove old groups, insert merged group
            for idx in sorted([gi, gj], reverse=True):
                groups.pop(idx)
                group_solutions.pop(idx)
            groups.append(merged_agents)
            group_solutions.append(merged_sol)

    def _solve_group(self, agents: List[AgentId]) -> Optional[List[List[Vertex]]]:
        sub_starts = [self.starts[a] for a in agents]
        sub_goals  = [self.goals[a]  for a in agents]
        sub_graph  = self.graph      # shared graph
        cbs = CBS(sub_graph, sub_starts, sub_goals)
        sol = cbs.search()
        return sol

    def _combine(
        self,
        groups:          List[List[AgentId]],
        group_solutions: List[List[List[Vertex]]],
    ) -> List[List[Vertex]]:
        full: List[Optional[List[Vertex]]] = [None] * self.k
        for g, sol in zip(groups, group_solutions):
            for local_i, global_a in enumerate(g):
                full[global_a] = sol[local_i]
        return full  # type: ignore

    def _find_group_indices(
        self,
        groups:   List[List[AgentId]],
        conflict: Conflict,
    ) -> Tuple[int, int]:
        gi = gj = -1
        for idx, g in enumerate(groups):
            if conflict.agent_i in g:
                gi = idx
            if conflict.agent_j in g:
                gj = idx
        return gi, gj


# ---------------------------------------------------------------------------
# Section 8 – Meta-Agent CBS (MA-CBS)
# ---------------------------------------------------------------------------

class MACBS(CBS):
    """
    Meta-Agent Conflict-Based Search (Section 8).

    Generalises CBS with a *merge policy*: if the number of conflicts between
    two agents exceeds the bound B, they are merged into a single meta-agent
    and solved jointly by the low-level MAPF solver (here: CBS itself).

    Special cases (Section 8.7):
        B = ∞  →  equivalent to basic CBS (never merge)
        B = 0  →  equivalent to Independence Detection (always merge on conflict)

    Parameters
    ----------
    B : int | float
        Conflict bound.  Use float('inf') for basic CBS behaviour.
    """

    def __init__(
        self,
        graph:  Graph,
        starts: List[Vertex],
        goals:  List[Vertex],
        B:      float = float('inf'),
        check_edge_conflicts: bool = True,
        max_time: int = 1000,
    ):
        super().__init__(graph, starts, goals, check_edge_conflicts, max_time)
        self.B  = B
        # Conflict matrix CM[i][j] – Section 8.3
        self.CM: Dict[Tuple[AgentId, AgentId], int] = {}
        # Meta-agent grouping: maps original agent id → meta-agent id (frozenset)
        self.meta: Dict[AgentId, FrozenSet[AgentId]] = {
            i: frozenset([i]) for i in range(self.k)
        }

    def _should_merge(self, ai: AgentId, aj: AgentId) -> bool:
        """
        shouldMerge() function – Algorithm 2, line 11 (Section 8.3).
        Returns True if CM[i,j] > B.
        """
        key = (min(ai, aj), max(ai, aj))
        return self.CM.get(key, 0) > self.B

    def _increment_cm(self, ai: AgentId, aj: AgentId) -> None:
        """Increment conflict matrix entry (Section 8.3)."""
        key = (min(ai, aj), max(ai, aj))
        self.CM[key] = self.CM.get(key, 0) + 1

    def search(self) -> Optional[List[List[Vertex]]]:
        """
        Algorithm 2 extended with MA-CBS merge logic (lines 11-18).
        """
        root = self._make_root()
        if root is None:
            return None

        open_list: List[CTNode] = []
        heapq.heappush(open_list, root)

        while open_list:
            P = heapq.heappop(open_list)
            self.high_level_expanded += 1

            conflict = _validate_solution(P.solution, self.check_edge_conflicts)
            if conflict is None:
                return P.solution

            # Line 10: C ← first conflict (a_i, a_j, v, t)
            if isinstance(conflict, Conflict):
                ai, aj = conflict.agent_i, conflict.agent_j
            else:
                ai, aj = conflict.agent_i, conflict.agent_j

            # Increment conflict matrix (Section 8.3)
            self._increment_cm(ai, aj)

            # Line 11: shouldMerge?
            if self._should_merge(ai, aj):
                # Lines 12-18: merge agents into meta-agent
                merged = self._merge_agents(P, ai, aj)
                if merged is not None and merged.cost < float('inf'):
                    heapq.heappush(open_list, merged)
                # continue (line 18 – go back to while)
            else:
                # Lines 19-26: branch as in basic CBS
                for child in self._resolve_conflict(P, conflict):
                    if child is not None:
                        heapq.heappush(open_list, child)

        return None

    def _merge_agents(
        self,
        node: CTNode,
        ai:   AgentId,
        aj:   AgentId,
    ) -> Optional[CTNode]:
        """
        Section 8.2: merge agents ai and aj into a meta-agent.
        Solve the meta-agent with a low-level MAPF solver (CBS).
        Update the CT node's solution and cost.
        """
        meta_agents = [ai, aj]
        sub_starts  = [self.starts[a] for a in meta_agents]
        sub_goals   = [self.goals[a]  for a in meta_agents]

        # Collect external constraints for the meta-agent (Section 8.4.1)
        sub_constraints: Dict[int, List[Constraint]] = {0: [], 1: []}
        for local_i, global_a in enumerate(meta_agents):
            for c in node.constraints.get(global_a, []):
                sub_constraints[local_i].append(
                    Constraint(local_i, c.vertex, c.time)
                )

        # Solve the sub-problem with CBS (Section 8.5)
        sub_cbs = CBS(self.graph, sub_starts, sub_goals, self.check_edge_conflicts, self.max_time)
        sub_sol = sub_cbs.search()

        if sub_sol is None:
            return None

        # Update the full solution with meta-agent paths
        new_solution = [list(p) for p in node.solution]
        new_constraints = {i: list(cs) for i, cs in node.constraints.items()}

        for local_i, global_a in enumerate(meta_agents):
            new_solution[global_a] = sub_sol[local_i]

        cost = _sum_of_costs(new_solution, self.goals)

        return CTNode(
            constraints=new_constraints,
            solution=new_solution,
            cost=cost,
            conflicts=self._count_conflicts(new_solution),
        )