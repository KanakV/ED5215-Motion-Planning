"""
CBS PLANNER  —  debug-panel patch
==================================
Add these two imports at the TOP of cbs_claude2.py (after your existing imports):

    from environment.atc_grid_env_central import cbs_debug

Then replace the final `return actions` in `cbs_planner()` with the block
shown in _PATCH_cbs_planner_return() below.

That's it. Nothing else in the planner changes.
"""

# ─── ADD to top of cbs_claude2.py ────────────────────────────────────────────

from environment.atc_grid_env_central import cbs_debug   # noqa: E402  (add after your imports)


# ─── REPLACE the tail of cbs_planner() with this ─────────────────────────────

def _PATCH_cbs_planner_return(
    paths,
    budget_exhausted,
    nodes_expanded,
    active_planes,
    runway,
    grid_size,
    remaining,      # result of _find_all_conflicts(paths, runway)
    actions,
    _step,
):
    """
    Drop-in replacement for the block that starts with
        _last_actions = dict(actions)
    at the bottom of cbs_planner().

    Copy-paste the body below over that existing block.
    """
    # ── Write into the shared debug singleton ────────────────────────────
    cbs_debug.step             = _step
    cbs_debug.grid_size        = grid_size
    cbs_debug.runway           = runway
    cbs_debug.warning_radius   = WARNING_RADIUS          # already imported

    # paths keyed by agent id, each path is list of (row,col) tuples
    cbs_debug.paths            = {pid: list(path) for pid, path in paths.items()}

    # current positions of all active agents
    cbs_debug.agent_starts     = {p["id"]: p["pos"] for p in active_planes}

    # conflicts: list of (a, b, pos, t, dist)
    cbs_debug.conflicts        = list(remaining)

    cbs_debug.nodes_expanded   = nodes_expanded
    cbs_debug.budget_exhausted = budget_exhausted
    cbs_debug.total_cost       = sum(len(p) - 1 for p in paths.values())

    # ── Push notable events to the rolling log ───────────────────────────
    if budget_exhausted:
        cbs_debug.push_log("error",
            f"Budget exhausted ({nodes_expanded} nodes, {len(active_planes)} agents)")

    imminent = [(a, b, loc, t, d) for a, b, loc, t, d in remaining if t <= 1]
    if imminent:
        for a, b, loc, t, dist in imminent:
            cbs_debug.push_log("error",
                f"IMMINENT conflict A{a}&A{b} t={t} d={dist:.2f}")
    elif remaining:
        cbs_debug.push_log("warn",
            f"{len(remaining)} conflict(s) in solution (non-imminent)")
    else:
        if active_planes:
            cbs_debug.push_log("ok",
                f"Clean solution — {nodes_expanded} nodes, cost={cbs_debug.total_cost}")

    # ── Duplicate-move check (your original TRACE 4) ─────────────────────
    nxt: dict = {}
    for pid, np_ in actions.items():
        nxt.setdefault(np_, []).append(pid)
    for np_, agent_list in nxt.items():
        if len(agent_list) > 1:
            cbs_debug.push_log("error", f"DUPE MOVE agents {agent_list}→{np_}")

    return actions


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE patched cbs_planner() tail — copy-paste this to replace the
# "Save state for next step diagnostics" block onwards in cbs_planner():
# ─────────────────────────────────────────────────────────────────────────────

PATCH_SNIPPET = """
    # ------------------------------------------------------------------
    # Write to CBS debug panel
    # ------------------------------------------------------------------
    cbs_debug.step             = _step
    cbs_debug.grid_size        = grid_size
    cbs_debug.runway           = runway
    cbs_debug.warning_radius   = WARNING_RADIUS

    cbs_debug.paths            = {pid: list(path) for pid, path in paths.items()}
    cbs_debug.agent_starts     = {p["id"]: p["pos"] for p in active_planes}
    cbs_debug.conflicts        = list(remaining)
    cbs_debug.nodes_expanded   = nodes_expanded
    cbs_debug.budget_exhausted = budget_exhausted
    cbs_debug.total_cost       = sum(len(p) - 1 for p in paths.values())

    if budget_exhausted:
        cbs_debug.push_log("error",
            f"Budget exhausted ({nodes_expanded} nodes, {len(active_planes)} agents)")
    imminent = [(a, b, loc, t, d) for a, b, loc, t, d in remaining if t <= 1]
    if imminent:
        for a, b, loc, t, dist in imminent:
            cbs_debug.push_log("error", f"IMMINENT A{a}&A{b} t={t} d={dist:.2f}")
    elif remaining:
        cbs_debug.push_log("warn", f"{len(remaining)} conflict(s) in solution")
    else:
        if active_planes:
            cbs_debug.push_log("ok",
                f"Clean — {nodes_expanded} nodes cost={cbs_debug.total_cost}")

    nxt: Dict[Pos, List[int]] = {}
    for pid, np_ in actions.items():
        nxt.setdefault(np_, []).append(pid)
    for np_, agent_list in nxt.items():
        if len(agent_list) > 1:
            cbs_debug.push_log("error", f"DUPE MOVE {agent_list}→{np_}")

    # ------------------------------------------------------------------
    # Save state for next-step diagnostics  (original lines — keep these)
    # ------------------------------------------------------------------
    _last_actions   = dict(actions)
    _last_positions = {p["id"]: p["pos"] for p in active_planes}
    _last_conflicts = remaining

    return actions
"""