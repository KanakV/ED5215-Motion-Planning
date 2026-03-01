# Conflict-Based Search (CBS) - No Wait Version

## Overview

This project implements **Conflict-Based Search (CBS)** for Multi-Agent
Pathfinding (MAPF) on a 2D grid.

### Key Characteristics

-   4-connected grid movement
-   Unit-cost moves
-   Static obstacles
-   Time-expanded planning
-   Vertex and Edge conflict handling
-   ❌ No wait action allowed

This implementation is suitable for systems where agents must
continuously move, such as: - Fixed-wing aircraft - Autonomous vehicles
without stopping capability - Continuous motion robotics

------------------------------------------------------------------------

# Algorithm Architecture

CBS is a two-level algorithm:

## 1. High-Level Search (Constraint Tree)

Each high-level node contains:

-   A set of constraints
-   A path for each agent
-   Total cost (sum of path lengths)

High-level search resolves conflicts by branching on constraints.

## 2. Low-Level Search (A\*)

For a single agent: - State = (position, time) - 4 directional moves
only - No waiting allowed - Constraints enforced during expansion

------------------------------------------------------------------------

# Conflict Types

## Vertex Conflict

Two agents occupy the same cell at the same time.

## Edge Conflict

Two agents swap positions at the same time step.

------------------------------------------------------------------------

# CBS Algorithm Flow

1.  Compute shortest path for each agent (ignoring conflicts).
2.  Detect the first conflict.
3.  If no conflict → return solution.
4.  Otherwise:
    -   Create two child nodes.
    -   Add a constraint to one agent in each child.
    -   Replan only that agent.
5.  Continue until conflict-free solution is found.

The priority queue is sorted by:

    Sum of path lengths

------------------------------------------------------------------------

# Important: No-Wait Implications

Since agents cannot wait:

-   Some problems become UNSOLVABLE.
-   Narrow corridors may cause deadlocks.
-   Agents may need to take longer detours.
-   Cyclic motion may appear.

This models realistic continuous-motion systems.

------------------------------------------------------------------------

# Data Structures

## Constraint Format

Vertex constraint:

    {
        "agent": id,
        "type": "vertex",
        "position": (x, y),
        "time": t
    }

Edge constraint:

    {
        "agent": id,
        "type": "edge",
        "from": (x1, y1),
        "to": (x2, y2),
        "time": t
    }

------------------------------------------------------------------------

# Complexity

Worst-case complexity is exponential in number of conflicts.

Low-level A\* complexity:

    O(|V| log |V|)

High-level complexity depends on number of generated constraint nodes.

------------------------------------------------------------------------

# How to Run

Example usage:

``` python
solution = cbs(agents, starts, goals, grid)
print(solution)
```

Grid format: - 0 = free cell - 1 = obstacle

------------------------------------------------------------------------

# Extending This Implementation

You can modify:

-   Heuristic → Euclidean or energy-aware
-   Grid → Continuous coordinates
-   Add altitude dimension
-   Replace Manhattan with domain-specific metric

------------------------------------------------------------------------

# Suggested Project Structure

    cbs/
        high_level.py
        low_level.py
        constraints.py
        conflict_detection.py

------------------------------------------------------------------------

# Notes for Motion Planning Projects

This implementation is ideal for:

-   Multi-aircraft landing sequencing
-   Warehouse robotics
-   Autonomous vehicle coordination

If using in air-traffic control simulation: - Replace obstacle logic
with no-fly zones. - Add separation buffers. - Consider velocity-based
constraints.

------------------------------------------------------------------------

# Conclusion

This is a clean, complete implementation of CBS without wait actions.

It is: - Optimal (sum-of-costs) - Complete (if solution exists under
no-wait rules) - Modular and extensible