"""
scaling_experiment.py
=====================
Runs all three algorithms headlessly across an increasing number of planes.
For each (N, seed) combination a fresh Scenario is generated, all three
algorithms simulate it, and metrics are collected.

Results are written to:
  results/scaling_raw.csv        — one row per (algorithm, N, seed)
  results/scaling_summary.csv    — mean ± std per (algorithm, N)
  results/scaling_plots.png      — grid of metric-vs-N plots

Configuration
-------------
Edit the block below to change the sweep range, seeds, or which metrics
to plot.
"""

import os
import sys
import csv
import json
import time
import math
import random
import itertools
import statistics

import numpy as np
import matplotlib

from algorithms.cbs_claude import cbs_planner
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config.config import SIM_TIME, GRID_SIZE, COLLISION_RADIUS, WARNING_RADIUS
from environment.atc_grid_env_central import Scenario, AlgoSimulation
# from algos.cooperative_astar import cooperative_planner
# from algos.longrange_astar   import lra_planner
# from algos.cbs_point         import cbs_planner

from planner2 import cooperative_planner
from planner3 import lra_planner
from planner4 import cbs_planner
# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT CONFIGURATION  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

PLANE_COUNTS = list(range(2, 20))   # e.g. [2, 3, 4, … 15]
N_SEEDS      = 3                    # random scenarios per N value
RESULTS_DIR  = os.path.join(ROOT, "results")

ALGORITHMS = [
    ("Cooperative",  cooperative_planner),
    ("Long Range A*", lra_planner),
    ("CBS",           cbs_planner),
]

# Metrics pulled from get_full_metrics() for the CSV and plots.
# Each entry: (display_label, nested_key_path)
# key_path is a dot-separated string into the metrics dict.
SCALAR_METRICS = [
    ("Completion Rate",              "completion.completion_rate"),
    ("Makespan (s)",                 "efficiency.makespan_s"),
    ("Avg Travel Time (s)",          "efficiency.avg_travel_time_s"),
    ("Avg Detour Ratio",             "normalised.avg_detour_ratio"),
    ("Avg Norm Travel Time",         "normalised.avg_norm_travel_time"),
    ("Collision Count",              "safety.collision_count"),
    ("Near Miss Count",              "safety.near_miss_count"),
    ("Safety Violation Rate",        "safety.safety_violation_rate"),
    ("Avg Planning Time/Agent (s)",  "normalised.avg_planning_time_per_agent_s"),
    ("P95 Planning Time (s)",        "computational.p95_planning_time_s"),
    ("Avg Plan Time Spawn Step (s)", "computational.avg_planning_time_spawn_step_s"),
    ("Avg Wait Ratio",               "normalised.avg_wait_ratio"),
    ("Detour Ratio CV",              "normalised.detour_ratio_cv"),
    ("Throughput (planes/s)",        "throughput.planes_per_second"),
    ("Avg Direction Changes",        "path_quality.avg_direction_changes"),
]

# Subset of metrics to show in the summary plot (max ~9 for readability)
PLOT_METRICS = [
    "Completion Rate",
    "Makespan (s)",
    "Avg Detour Ratio",
    "Collision Count",
    "Avg Planning Time/Agent (s)",
    "P95 Planning Time (s)",
    "Avg Wait Ratio",
    "Detour Ratio CV",
    "Throughput (planes/s)",
]

ALGO_COLORS = {
    "Cooperative":   "#4C72B0",
    "Long Range A*": "#DD8452",
    "CBS":           "#55A868",
}

# ─────────────────────────────────────────────────────────────────────────────


def get_nested(d: dict, dotpath: str):
    """Retrieve a value from a nested dict using a dot-separated key path."""
    for key in dotpath.split("."):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def run_headless(scenario: Scenario, planner, name: str) -> dict:
    """
    Simulate a single algorithm on a scenario without any visualiser.
    Steps are driven by a simple time loop matching the visualiser's cadence.
    Returns the full metrics dict.
    """
    sim = AlgoSimulation(scenario, planner, name)

    dt          = 0.3          # seconds per step (matches visualiser interval=300ms)
    current_t   = 0.0
    total_steps = int(SIM_TIME / dt) + 1

    for _ in range(total_steps):
        sim.step(current_t)
        current_t += dt

    sim.metrics.finish()
    return sim.get_full_metrics()


def run_experiment() -> list[dict]:
    """
    Main loop: sweep over PLANE_COUNTS × N_SEEDS × ALGORITHMS.
    Returns a list of flat result rows ready for CSV export.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows        = []
    total_runs  = len(PLANE_COUNTS) * N_SEEDS * len(ALGORITHMS)
    run_idx     = 0

    for n_planes in PLANE_COUNTS:
        for seed in range(N_SEEDS):

            # All algorithms share the same scenario for this (N, seed) pair
            random.seed(seed * 1000 + n_planes)
            np.random.seed(seed * 1000 + n_planes)

            scenario = Scenario(
                grid_size  = GRID_SIZE,
                max_planes = n_planes,
                sim_time   = SIM_TIME,
            )

            for algo_name, planner in ALGORITHMS:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}]  N={n_planes:>3}  seed={seed}  {algo_name}")

                t_start = time.time()
                metrics = run_headless(scenario, planner, algo_name)
                elapsed = time.time() - t_start

                row = {
                    "algorithm": algo_name,
                    "n_planes":  n_planes,
                    "seed":      seed,
                    "wall_time_s": round(elapsed, 3),
                }

                for label, keypath in SCALAR_METRICS:
                    val = get_nested(metrics, keypath)
                    row[label] = round(val, 6) if isinstance(val, float) else val

                rows.append(row)

    return rows


def save_raw_csv(rows: list[dict]):
    path = os.path.join(RESULTS_DIR, "scaling_raw.csv")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nRaw results  → {path}")


def save_summary_csv(rows: list[dict]):
    """Aggregate rows by (algorithm, n_planes): compute mean and std per metric."""
    from collections import defaultdict

    # Group by (algo, n)
    groups = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], row["n_planes"])
        groups[key].append(row)

    summary_rows = []
    metric_labels = [label for label, _ in SCALAR_METRICS]

    for (algo, n), group in sorted(groups.items()):
        srow = {"algorithm": algo, "n_planes": n}
        for label in metric_labels:
            vals = [r[label] for r in group if r[label] is not None]
            if vals:
                mean = statistics.mean(vals)
                std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
                srow[f"{label}_mean"] = round(mean, 6)
                srow[f"{label}_std"]  = round(std,  6)
            else:
                srow[f"{label}_mean"] = None
                srow[f"{label}_std"]  = None
        summary_rows.append(srow)

    path = os.path.join(RESULTS_DIR, "scaling_summary.csv")
    fieldnames = list(summary_rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Summary CSV  → {path}")
    return summary_rows


def save_plots(summary_rows: list[dict]):
    """
    One subplot per metric in PLOT_METRICS.
    Each subplot shows mean ± std shading per algorithm vs N.
    """
    from collections import defaultdict

    # Restructure: data[algo][metric][n] = (mean, std)
    data = defaultdict(lambda: defaultdict(dict))
    for row in summary_rows:
        algo = row["algorithm"]
        n    = row["n_planes"]
        for label in PLOT_METRICS:
            mean = row.get(f"{label}_mean")
            std  = row.get(f"{label}_std", 0.0)
            if mean is not None:
                data[algo][label][n] = (mean, std or 0.0)

    n_plots = len(PLOT_METRICS)
    ncols   = 3
    nrows   = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             constrained_layout=True)
    fig.suptitle("Algorithm Scaling — Metrics vs Number of Planes",
                 fontsize=14, fontweight="bold", y=1.01)

    axes_flat = axes.flatten() if n_plots > 1 else [axes]

    for ax, metric_label in zip(axes_flat, PLOT_METRICS):
        for algo_name, _ in ALGORITHMS:
            color    = ALGO_COLORS.get(algo_name, "grey")
            ns       = sorted(data[algo_name][metric_label].keys())
            means    = [data[algo_name][metric_label][n][0] for n in ns]
            stds     = [data[algo_name][metric_label][n][1] for n in ns]

            means_arr = np.array(means)
            stds_arr  = np.array(stds)

            ax.plot(ns, means_arr, label=algo_name, color=color,
                    linewidth=2, marker="o", markersize=4)
            ax.fill_between(ns,
                            means_arr - stds_arr,
                            means_arr + stds_arr,
                            alpha=0.15, color=color)

        ax.set_title(metric_label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Number of Planes (N)")
        ax.set_xticks(PLANE_COUNTS)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    # Add a shared legend on the first axis
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(ALGORITHMS),
               bbox_to_anchor=(0.5, -0.03), fontsize=10,
               frameon=False)

    # Hide any unused subplots
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    path = os.path.join(RESULTS_DIR, "scaling_plots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots        → {path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" Multi-Agent Path Planning — Scaling Experiment")
    print(f" Plane counts : {PLANE_COUNTS}")
    print(f" Seeds / N    : {N_SEEDS}")
    print(f" Algorithms   : {[a for a, _ in ALGORITHMS]}")
    print(f" Total runs   : {len(PLANE_COUNTS) * N_SEEDS * len(ALGORITHMS)}")
    print("=" * 60)

    t0   = time.time()
    rows = run_experiment()

    print(f"\nAll runs complete in {time.time() - t0:.1f}s")
    print("Saving outputs …")

    save_raw_csv(rows)
    summary = save_summary_csv(rows)
    save_plots(summary)

    print("\nDone.")