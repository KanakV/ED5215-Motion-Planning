# Metrics Reference — Multi-Agent Path Planning Simulator

This document describes every metric tracked by `MetricsTracker` across all algorithm simulations. Metrics are grouped by category. Each entry states what is measured, how it is computed, what a good value looks like, and when to pay attention to it.

---

## How Metrics Are Recorded

The simulator runs a shared `Scenario` (same runway position, same spawn schedule) across all algorithms simultaneously. At each step, `MetricsTracker` receives:

- The real simulation time (`current_time` in seconds)
- The planner's wall-clock compute time for that step
- The list of currently active planes and their updated positions
- A flag indicating whether a new plane spawned this step

Travel times are computed in **real simulation-time seconds**, not step indices. Each plane's timer starts the moment it spawns (`register_spawn`) and stops the moment it lands (`record_landing`). This correctly handles staggered spawns — a plane that spawns at `t=1.8` and lands at `t=9.2` has a travel time of `7.4s`, regardless of when other planes entered the arena.

---

## Categories

1. [Efficiency](#1-efficiency)
2. [Safety](#2-safety)
3. [Computational](#3-computational)
4. [Congestion](#4-congestion)
5. [Fairness](#5-fairness)
6. [Throughput](#6-throughput)
7. [Path Quality](#7-path-quality)
8. [Completion](#8-completion)
9. [Normalised Metrics](#9-normalised-metrics)

---

## 1. Efficiency

Measures how well agents find short, fast paths to the runway.

| Metric | Unit | Description |
|---|---|---|
| `total_path_length` | grid units | Sum of Euclidean path lengths across all planes |
| `avg_path_length` | grid units | `total_path_length / n_planes` |
| `makespan_s` | seconds | Time from the first plane's spawn to the last plane's landing |
| `avg_travel_time_s` | seconds | Mean of per-plane travel times (spawn-to-land, stagger-corrected) |
| `travel_time_per_plane_s` | seconds | Per-plane breakdown of travel time |

**Makespan** is the standard MAPF primary metric — minimising it means the whole fleet clears as fast as possible.

**avg_travel_time_s** captures aggregate cost. An algorithm could have good makespan but poor average travel time if it sacrifices some agents to fast-track others.

---

## 2. Safety

Measures how often planes come dangerously close to each other.

| Metric | Unit | Description |
|---|---|---|
| `collision_count` | count | Number of unique pairs that entered the collision radius in the same step |
| `near_miss_count` | count | Number of unique pairs that entered the warning radius (but not collision radius) |
| `min_separation_distance` | grid units | Closest any two planes ever were across the entire run |
| `safety_violation_rate` | ratio [0, 1] | Fraction of all pair-steps where separation was below the collision threshold |

**Collision radius** and **warning radius** are set in `config.py` as `COLLISION_RADIUS` and `WARNING_RADIUS`.

`safety_violation_rate` is the most comparable safety metric across runs with different numbers of planes, since it normalises by the total number of pair-steps observed.

`min_separation_distance` is useful as a worst-case signal even when collision count is zero.

---

## 3. Computational

Measures the planning overhead of each algorithm.

| Metric | Unit | Description |
|---|---|---|
| `max_planning_time_s` | seconds | Single slowest planning call across the entire run |
| `p95_planning_time_s` | seconds | 95th percentile planning time — robust worst-case estimate |
| `std_planning_time_s` | seconds | Standard deviation of planning times — measures consistency |
| `total_runtime_s` | seconds | Total wall-clock time from `metrics.start()` to `metrics.finish()` |
| `avg_planning_time_spawn_step_s` | seconds | Average planning time on steps where a new plane spawned |
| `avg_planning_time_non_spawn_step_s` | seconds | Average planning time on all other steps |

**Why P95 over max?** Max planning time is often a one-off spike (e.g. a GC pause or OS interrupt). P95 gives a stable picture of the algorithm's actual worst-case behaviour.

**Spawn-step vs non-spawn-step planning time** isolates the dynamic replanning cost. If `avg_planning_time_spawn_step_s` is significantly higher, the algorithm struggles to integrate new agents into existing plans. This is the most operationally relevant stress test for a centralised planner.

---

## 4. Congestion

Measures how often planes are stuck waiting instead of moving.

| Metric | Unit | Description |
|---|---|---|
| `avg_active_agents_per_step` | count | Mean number of planes in the arena per step |
| `avg_wait_time_steps` | steps | Mean wait steps per plane across all planes |
| `total_wait_steps` | steps | Sum of wait steps across all planes |

A **wait step** is recorded when a plane's position does not change between steps. Lateral movements (circling to avoid conflict) are **not** counted as waits — only literal non-movement is.

High wait time with low collision count often means the algorithm is being overly conservative, holding planes back unnecessarily.

---

## 5. Fairness

Measures whether the algorithm treats all planes equitably, regardless of where they spawn.

| Metric | Unit | Description |
|---|---|---|
| `travel_time_variance` | seconds² | Variance of per-plane travel times across all landed planes |

High variance means some planes land quickly while others are delayed disproportionately. This can be a spawn-position artefact (planes that spawn far from the runway naturally take longer), which is why the normalised counterpart `detour_ratio_cv` is more informative for algorithm comparison.

---

## 6. Throughput

Measures the rate at which the fleet is cleared.

| Metric | Unit | Description |
|---|---|---|
| `planes_per_second` | planes/s | Number of planes that landed divided by total wall-clock runtime |

Throughput complements makespan. An algorithm with a slightly worse makespan may still have higher throughput if it lands planes more evenly throughout the run rather than in a burst at the end. This distinction becomes important when scaling `N`.

---

## 7. Path Quality

Measures the smoothness of the paths produced by the planner.

| Metric | Unit | Description |
|---|---|---|
| `avg_direction_changes` | count | Mean number of direction changes per plane |
| `direction_changes_per_plane` | count | Per-plane breakdown |

A **direction change** is recorded when a plane's movement vector differs from the previous step's movement vector. Two algorithms with identical `avg_detour_ratio` can be distinguished here — one may produce smooth, curved paths while the other produces erratic zig-zagging paths that would be unrealistic in a real ATC context.

---

## 8. Completion

Measures whether all spawned planes actually made it to the runway.

| Metric | Unit | Description |
|---|---|---|
| `planes_spawned` | count | Total planes that entered the arena |
| `planes_landed` | count | Total planes that reached the runway before `sim_time` elapsed |
| `completion_rate` | ratio [0, 1] | `planes_landed / planes_spawned` |

This metric becomes critical when scaling `N`. At low agent counts, all algorithms will likely achieve `completion_rate = 1.0`. As `N` grows, weaker algorithms will start timing out with planes still airborne. Any comparison of efficiency or safety metrics between algorithms with different completion rates is misleading — always check this first.

---

## 9. Normalised Metrics

These metrics adjust for differences in spawn distance, fleet size, and problem difficulty to allow fair cross-algorithm and cross-configuration comparison.

| Metric | Formula | Range | Interpretation |
|---|---|---|---|
| `avg_detour_ratio` | `path_length / manhattan_to_runway` | ≥ 1.0 | 1.0 = flew the shortest possible path. 1.4 = flew 40% further than optimal |
| `detour_ratio_per_plane` | per-plane detour ratios | ≥ 1.0 | Identify which planes were most penalised |
| `avg_norm_travel_time` | `travel_time_s / manhattan_to_runway` | ≥ 0 | Seconds spent per unit of Manhattan distance. Accounts for planes spawning at different distances from the runway |
| `norm_travel_time_per_plane` | per-plane normalised travel times | ≥ 0 | Per-plane breakdown |
| `avg_planning_time_per_agent_s` | `planning_time_s / n_active_agents` | ≥ 0 | Isolates algorithm cost from fleet size. Use this when comparing algorithms at different values of N |
| `max_planning_time_per_agent_s` | max over all steps | ≥ 0 | Worst-case per-agent planning burden |
| `avg_wait_ratio` | `wait_steps / travel_steps` per plane, averaged | [0, 1] | 0.0 = no plane ever waited. 1.0 = every plane spent its entire trip stationary |
| `wait_ratio_per_plane` | per-plane wait ratios | [0, 1] | Identify planes that were disproportionately held |
| `detour_ratio_cv` | `std(detour_ratios) / mean(detour_ratios)` | ≥ 0 | Coefficient of variation. 0.0 = all planes got equally efficient paths. Higher = algorithm is unfair to certain spawn positions |

### Why normalise?

Raw metrics conflate algorithm quality with problem geometry. A plane spawning in the corner of the grid has a longer Manhattan distance to the runway than one spawning on the nearest border edge. Without normalisation, algorithms that happen to receive favourably-positioned spawns in a random scenario will appear better than they are. Normalised metrics control for this and remain valid across different grid sizes, runway positions, and agent counts.

---

## Output Files

| File | Contents |
|---|---|
| `results/metrics.json` | Full nested metrics dict for all algorithms, including per-plane breakdowns |
| `results/metrics_summary.csv` | Flat one-row-per-algorithm summary of all scalar metrics |
| `logs/<algorithm>_replay.json` | Step-by-step position log for visualisation replay |

---

## Metric Priority for Algorithm Comparison

When comparing algorithms, evaluate in this order:

1. **`completion_rate`** — disqualifies any comparison where one algorithm failed to land all planes
2. **`collision_count` / `safety_violation_rate`** — safety is non-negotiable
3. **`avg_detour_ratio`** — primary efficiency signal, geometry-corrected
4. **`avg_norm_travel_time`** — time efficiency, spawn-stagger corrected
5. **`avg_planning_time_per_agent_s`** — scalability signal
6. **`detour_ratio_cv`** — fairness across spawn positions
7. **`avg_wait_ratio`** — congestion / conservatism signal
8. **`avg_direction_changes`** — path realism / smoothness