import numpy as np
import random
import math
import time
import csv
import json
import os
import statistics
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D
from matplotlib.patches import Circle
from config.config import PLANE_RADIUS, COLLISION_RADIUS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
RESULTS_DIR = os.path.join(PARENT_DIR, "results")
SIMULATION_LOG_DIR = os.path.join(RESULTS_DIR, "simulation")

# Ensure output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SIMULATION_LOG_DIR, exist_ok=True)

print(BASE_DIR)


# =====================================================
# SCENARIO (Shared Random Scenario Per Run)
# =====================================================

class Scenario:

    def __init__(self, grid_size, max_planes, sim_time):
        self.grid_size = grid_size
        self.sim_time = sim_time

        # Random runway
        self.runway = (
            random.randint(grid_size // 4, 3 * grid_size // 4),
            random.randint(grid_size // 4, 3 * grid_size // 4)
        )

        # Pre-generate spawn events (time + position)
        self.spawn_events = []

        t = 0
        count = 0

        while t < sim_time and count < max_planes:
            t += random.uniform(0.5, 4)

            border_positions = []
            for i in range(grid_size):
                border_positions.append((0, i))
                border_positions.append((grid_size - 1, i))
                border_positions.append((i, 0))
                border_positions.append((i, grid_size - 1))

            pos = random.choice(border_positions)
            self.spawn_events.append((t, pos))
            count += 1


# =====================================================
# METRICS TRACKER
# =====================================================

class MetricsTracker:
    """
    Tracks all metrics per simulation:
      Efficiency   : path length per plane, total path length, makespan, avg travel time
      Safety       : collision count, near miss count, min separation, safety violation rate
      Computational: max planning time per step, std dev of planning time, total runtime
      Congestion   : avg active agents per step, wait time per plane
      Fairness     : variance of travel time
    """

    def __init__(self, collision_threshold, warning_threshold):
        self.collision_threshold = collision_threshold
        self.warning_threshold = warning_threshold

        # Per-step records
        self.planning_times = []           # compute time each step
        self.active_counts = []            # active plane count each step

        # Per-plane records  {plane_id: {...}}
        self.plane_spawn_step = {}         # step when plane spawned
        self.plane_land_step = {}          # step when plane landed
        self.plane_wait_steps = {}         # steps where plane didn't move toward runway

        # Safety counters
        self.collision_count = 0
        self.near_miss_count = 0
        self.min_separation = float("inf")
        self.total_pair_steps = 0          # total (active pair, step) combinations
        self.violation_steps = 0           # pair-steps inside collision threshold

        # Timing
        self.wall_start = None
        self.wall_end = None

        self._step_index = 0
        self._collisions_this_step = set()
        self._near_misses_this_step = set()

    # --------------------------------------------------

    def start(self):
        self.wall_start = time.time()

    def finish(self):
        self.wall_end = time.time()

    # --------------------------------------------------

    def record_step(self, planning_time, active_planes, prev_positions):
        """
        Call once per simulation step AFTER moves have been applied.

        active_planes  : list of active plane dicts (with updated pos)
        prev_positions : dict {plane_id: pos_before_move}
        """
        self._step_index += 1
        self.planning_times.append(planning_time)
        self.active_counts.append(len(active_planes))

        # --- Spawn / land tracking ---
        for p in active_planes:
            pid = p["id"]
            if pid not in self.plane_spawn_step:
                self.plane_spawn_step[pid] = self._step_index

        # --- Wait time: did the plane move closer to runway? ---
        # (tracked externally via record_landing / record_wait)

        # --- Safety checks ---
        self._collisions_this_step = set()
        self._near_misses_this_step = set()

        for i in range(len(active_planes)):
            for j in range(i + 1, len(active_planes)):
                p1 = active_planes[i]
                p2 = active_planes[j]
                dist = float(np.linalg.norm(
                    np.array(p1["pos"]) - np.array(p2["pos"])
                ))

                if dist < self.min_separation:
                    self.min_separation = dist

                self.total_pair_steps += 1

                if dist < self.collision_threshold:
                    self.violation_steps += 1
                    pair = (min(p1["id"], p2["id"]), max(p1["id"], p2["id"]))
                    if pair not in self._collisions_this_step:
                        self._collisions_this_step.add(pair)
                        self.collision_count += 1

                elif dist < self.warning_threshold:
                    pair = (min(p1["id"], p2["id"]), max(p1["id"], p2["id"]))
                    if pair not in self._near_misses_this_step:
                        self._near_misses_this_step.add(pair)
                        self.near_miss_count += 1

    # --------------------------------------------------

    def record_landing(self, plane_id):
        self.plane_land_step[plane_id] = self._step_index

    def record_wait(self, plane_id):
        self.plane_wait_steps[plane_id] = self.plane_wait_steps.get(plane_id, 0) + 1

    # --------------------------------------------------

    def compute(self, planes):
        """
        Returns a dict of all metrics given the final list of plane dicts.
        """
        # --- Efficiency ---
        path_lengths = {p["id"]: p["path_length"] for p in planes}
        total_path = sum(path_lengths.values())
        avg_path = total_path / len(planes) if planes else 0

        # Travel time = steps from spawn to landing (only landed planes)
        travel_times = []
        for pid, land in self.plane_land_step.items():
            spawn = self.plane_spawn_step.get(pid, 1)
            travel_times.append(land - spawn)

        makespan = max(self.plane_land_step.values()) - min(self.plane_spawn_step.values()) \
            if self.plane_land_step else 0
        avg_travel_time = statistics.mean(travel_times) if travel_times else 0

        # --- Safety ---
        safety_violation_rate = (
            self.violation_steps / self.total_pair_steps
            if self.total_pair_steps > 0 else 0
        )
        min_sep = self.min_separation if self.min_separation != float("inf") else None

        # --- Computational ---
        max_planning = max(self.planning_times) if self.planning_times else 0
        std_planning = statistics.stdev(self.planning_times) if len(self.planning_times) > 1 else 0
        total_runtime = (self.wall_end - self.wall_start) if (self.wall_start and self.wall_end) else 0

        # --- Congestion ---
        avg_active = statistics.mean(self.active_counts) if self.active_counts else 0
        total_wait = sum(self.plane_wait_steps.values())
        avg_wait = total_wait / len(planes) if planes else 0

        # --- Fairness ---
        travel_time_variance = statistics.variance(travel_times) if len(travel_times) > 1 else 0

        return {
            "efficiency": {
                "path_length_per_plane": path_lengths,
                "total_path_length": total_path,
                "avg_path_length": avg_path,
                "makespan_steps": makespan,
                "avg_travel_time_steps": avg_travel_time,
            },
            "safety": {
                "collision_count": self.collision_count,
                "near_miss_count": self.near_miss_count,
                "min_separation_distance": min_sep,
                "safety_violation_rate": safety_violation_rate,
            },
            "computational": {
                "max_planning_time_s": max_planning,
                "std_planning_time_s": std_planning,
                "total_runtime_s": total_runtime,
            },
            "congestion": {
                "avg_active_agents_per_step": avg_active,
                "avg_wait_time_steps": avg_wait,
                "total_wait_steps": total_wait,
            },
            "fairness": {
                "travel_time_variance": travel_time_variance,
                "travel_times_per_plane": {
                    pid: self.plane_land_step[pid] - self.plane_spawn_step.get(pid, 1)
                    for pid in self.plane_land_step
                },
            },
        }


# =====================================================
# SINGLE ALGORITHM SIMULATION
# =====================================================

class AlgoSimulation:

    def __init__(self, scenario: Scenario, planner, name: str):
        self.name = name
        self.planner = planner

        self.grid_size = scenario.grid_size
        self.runway = scenario.runway
        self.spawn_events = scenario.spawn_events.copy()

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self.plane_radius = PLANE_RADIUS
        self.collision_threshold = COLLISION_RADIUS
        self.warning_threshold = PLANE_RADIUS * 2.5

        self.planes = []
        self.plane_counter = 0

        self.total_compute_time = 0
        self.planning_calls = 0

        # Metrics
        self.metrics = MetricsTracker(
            collision_threshold=self.collision_threshold,
            warning_threshold=self.warning_threshold,
        )
        self.metrics.start()

        # Replay log
        self._replay_log = {
            "algorithm": name,
            "grid_size": self.grid_size,
            "runway": list(self.runway),
            "steps": [],
        }

    # -------------------------------------------------

    def step(self, current_time):

        # Spawn using shared schedule
        while self.spawn_events and current_time >= self.spawn_events[0][0]:
            _, pos = self.spawn_events.pop(0)

            plane = {
                "id": self.plane_counter,
                "pos": pos,
                "prev_pos": pos,
                "history": [pos],
                "active": True,
                "path_length": 0,
                "color": np.random.rand(3,).tolist(),
            }

            self.plane_counter += 1
            self.planes.append(plane)

        # Collect active planes
        active_planes = [p for p in self.planes if p["active"]]

        if not active_planes:
            return

        # Snapshot positions before move
        prev_positions = {p["id"]: p["pos"] for p in active_planes}

        # CENTRALIZED PLANNING CALL
        start_compute = time.time()

        actions = self.planner(
            self.grid.copy(),
            active_planes,
            self.runway
        )

        compute = time.time() - start_compute
        self.total_compute_time += compute
        self.planning_calls += 1

        # Execute actions simultaneously
        newly_landed = []

        for plane in active_planes:

            if plane["id"] not in actions:
                # Plane didn't move — record wait
                self.metrics.record_wait(plane["id"])
                continue

            next_pos = actions[plane["id"]]

            # Did the plane move closer to runway?
            dist_before = np.linalg.norm(
                np.array(plane["pos"]) - np.array(self.runway)
            )
            dist_after = np.linalg.norm(
                np.array(next_pos) - np.array(self.runway)
            )
            if dist_after >= dist_before:
                self.metrics.record_wait(plane["id"])

            plane["prev_pos"] = plane["pos"]
            plane["path_length"] += float(np.linalg.norm(
                np.array(next_pos) - np.array(plane["pos"])
            ))
            plane["pos"] = next_pos
            plane["history"].append(next_pos)

            if plane["pos"] == self.runway:
                plane["active"] = False
                newly_landed.append(plane["id"])

        # Record landings before metrics step
        for pid in newly_landed:
            self.metrics.record_landing(pid)

        # Record metrics for this step
        active_after = [p for p in self.planes if p["active"]]
        self.metrics.record_step(compute, active_after, prev_positions)

        # --- Replay log entry ---
        self._replay_log["steps"].append({
            "sim_time": current_time,
            "planes": [
                {
                    "id": p["id"],
                    "pos": list(p["pos"]),
                    "prev_pos": list(p["prev_pos"]),
                    "active": p["active"],
                    "path_length": p["path_length"],
                    "color": p["color"],
                }
                for p in self.planes
            ],
        })

        self._update_grid()

    # -------------------------------------------------

    def _update_grid(self):
        self.grid[:] = 0
        self.grid[self.runway] = 1

    # -------------------------------------------------

    def finalize(self):
        """Call after simulation ends to stop timers and save replay log."""
        self.metrics.finish()

        safe_name = self.name.replace(" ", "_").lower()
        log_path = os.path.join(SIMULATION_LOG_DIR, f"{safe_name}_replay.json")

        with open(log_path, "w") as f:
            json.dump(self._replay_log, f)

        print(f"[{self.name}] Replay log saved → {log_path}")

    # -------------------------------------------------

    def compute_statistics(self):

        rows = []
        total_length = 0

        for plane in self.planes:
            rows.append({
                "algorithm": self.name,
                "plane_id": plane["id"],
                "path_length": plane["path_length"]
            })
            total_length += plane["path_length"]

        avg_length = total_length / len(self.planes) if self.planes else 0
        avg_compute = self.total_compute_time / self.planning_calls if self.planning_calls else 0

        return rows, avg_length, avg_compute

    # -------------------------------------------------

    def get_full_metrics(self):
        return self.metrics.compute(self.planes)


# =====================================================
# MULTI PANEL VISUALIZER
# =====================================================

class MultiAlgorithmVisualizer:

    def __init__(self, scenario, simulations):

        self.scenario = scenario
        self.simulations = simulations

        self.fig, self.axes = plt.subplots(1, 3, figsize=(20, 6))
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, wspace=0.15)

        self.plane_img = plt.imread(os.path.join(RESOURCES_DIR, "plane.png"))
        self.runway_img = plt.imread(os.path.join(RESOURCES_DIR, "runway.png"))
        self.background_img = plt.imread(os.path.join(RESOURCES_DIR, "background.jpg"))

        self.start_time = None

    # -------------------------------------------------

    def run(self):

        self.start_time = time.time()

        self.ani = animation.FuncAnimation(
            self.fig,
            self._animate,
            interval=300
        )

        plt.show()
        self._finalize_all()

    # -------------------------------------------------

    def _draw_grid(self, ax, grid_size):

        for i in range(grid_size + 1):

            if i % 5 == 0:
                alpha = 0.08
                lw = 1
            else:
                alpha = 0.02
                lw = 0.5

            ax.axhline(i, color='black', alpha=alpha, linewidth=lw, zorder=1)
            ax.axvline(i, color='black', alpha=alpha, linewidth=lw, zorder=1)

    # -------------------------------------------------

    def _animate(self, frame):

        current_time = time.time() - self.start_time

        for sim in self.simulations:
            sim.step(current_time)

        for ax, sim in zip(self.axes, self.simulations):

            ax.clear()
            ax.set_xlim(0, sim.grid_size)
            ax.set_ylim(0, sim.grid_size)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

            ax.imshow(
                self.background_img,
                extent=[0, sim.grid_size, 0, sim.grid_size],
                zorder=0
            )

            self._draw_grid(ax, sim.grid_size)

            rx, ry = sim.runway
            ax.imshow(
                self.runway_img,
                extent=[rx - 8, rx + 8, ry - 3, ry + 3],
                zorder=2
            )

            # Trails
            for plane in sim.planes:
                if len(plane["history"]) > 1:
                    hx, hy = zip(*plane["history"])
                    ax.plot(hx, hy,
                            color=plane["color"],
                            linewidth=2,
                            alpha=0.8,
                            zorder=2)

            # Planes with rotation
            for plane in sim.planes:

                if not plane["active"]:
                    continue

                x, y = plane["pos"]
                px, py = plane["prev_pos"]

                dx = x - px
                dy = y - py

                angle = math.degrees(math.atan2(dy, dx)) - 90

                trans = Affine2D().rotate_deg_around(x, y, angle) + ax.transData

                ax.imshow(
                    self.plane_img,
                    extent=[
                        x - sim.plane_radius,
                        x + sim.plane_radius,
                        y - sim.plane_radius,
                        y + sim.plane_radius
                    ],
                    transform=trans,
                    zorder=3
                )

            # Proximity & Collision visualization
            for i in range(len(sim.planes)):
                for j in range(i + 1, len(sim.planes)):

                    p1 = sim.planes[i]
                    p2 = sim.planes[j]

                    if not p1["active"] or not p2["active"]:
                        continue

                    dist = np.linalg.norm(
                        np.array(p1["pos"]) - np.array(p2["pos"])
                    )

                    warning_threshold = sim.plane_radius * 2.5
                    collision_threshold = sim.plane_radius * 1.0

                    if dist < collision_threshold:
                        ax.add_patch(
                            Circle(
                                p1["pos"],
                                sim.plane_radius * 1.2,
                                facecolor="red",
                                edgecolor="darkred",
                                alpha=0.6,
                                zorder=5
                            )
                        )
                        ax.add_patch(
                            Circle(
                                p2["pos"],
                                sim.plane_radius * 1.2,
                                facecolor="red",
                                edgecolor="darkred",
                                alpha=0.6,
                                zorder=5
                            )
                        )

                    elif dist < warning_threshold:
                        ax.add_patch(
                            Circle(
                                p1["pos"],
                                sim.plane_radius * 1.2,
                                fill=False,
                                edgecolor="red",
                                linewidth=2,
                                zorder=4
                            )
                        )
                        ax.add_patch(
                            Circle(
                                p2["pos"],
                                sim.plane_radius * 1.2,
                                fill=False,
                                edgecolor="red",
                                linewidth=2,
                                zorder=4
                            )
                        )

            ax.set_title(sim.name)

            ax.text(
                sim.grid_size * 0.98,
                sim.grid_size * 0.02,
                f"{current_time:.1f}s",
                color="white",
                fontsize=10,
                ha="right",
                va="bottom",
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
            )

        return []

    # -------------------------------------------------

    def _finalize_all(self):
        """Finalize all simulations, save replay logs, export metrics."""
        for sim in self.simulations:
            sim.finalize()

        self._export_results()
        self._export_metrics()

    # -------------------------------------------------

    def _export_results(self):

        print("\n==== FINAL RESULTS ====\n")

        algo_names = [sim.name for sim in self.simulations]
        plane_count = max(len(sim.planes) for sim in self.simulations)

        table = []

        for i in range(plane_count):
            row = {"Plane": i + 1}

            for sim in self.simulations:
                if i < len(sim.planes):
                    row[sim.name] = sim.planes[i]["path_length"]
                else:
                    row[sim.name] = ""

            table.append(row)

        avg_row = {"Plane": "avg"}

        for sim in self.simulations:
            total = sum(p["path_length"] for p in sim.planes)
            avg = total / len(sim.planes) if sim.planes else 0
            avg_row[sim.name] = avg
            print(f"{sim.name} Average Path Length: {avg:.3f}")

        table.append(avg_row)

        fieldnames = ["Plane"] + algo_names
        csv_path = os.path.join(PARENT_DIR, "algorithm_comparison_table.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table)

        print("\nComparison table saved to algorithm_comparison_table.csv")

    # -------------------------------------------------

    def _export_metrics(self):
        """
        Saves a comprehensive metrics JSON and a flat summary CSV
        to results/metrics.json and results/metrics_summary.csv.
        """
        print("\n==== METRICS ====\n")

        all_metrics = {}

        for sim in self.simulations:
            m = sim.get_full_metrics()
            all_metrics[sim.name] = m

            print(f"\n--- {sim.name} ---")

            eff = m["efficiency"]
            print(f"  Total Path Length       : {eff['total_path_length']:.3f}")
            print(f"  Avg Path Length         : {eff['avg_path_length']:.3f}")
            print(f"  Makespan (steps)        : {eff['makespan_steps']}")
            print(f"  Avg Travel Time (steps) : {eff['avg_travel_time_steps']:.2f}")

            saf = m["safety"]
            print(f"  Collisions              : {saf['collision_count']}")
            print(f"  Near Misses             : {saf['near_miss_count']}")
            min_sep = saf['min_separation_distance']
            print(f"  Min Separation          : {min_sep:.3f}" if min_sep is not None else "  Min Separation          : N/A")
            print(f"  Safety Violation Rate   : {saf['safety_violation_rate']:.4f}")

            comp = m["computational"]
            print(f"  Max Planning Time (s)   : {comp['max_planning_time_s']:.6f}")
            print(f"  Std Planning Time (s)   : {comp['std_planning_time_s']:.6f}")
            print(f"  Total Runtime (s)       : {comp['total_runtime_s']:.3f}")

            cong = m["congestion"]
            print(f"  Avg Active Agents/Step  : {cong['avg_active_agents_per_step']:.2f}")
            print(f"  Avg Wait Time (steps)   : {cong['avg_wait_time_steps']:.2f}")

            fair = m["fairness"]
            print(f"  Travel Time Variance    : {fair['travel_time_variance']:.2f}")

        # Save full JSON
        json_path = os.path.join(RESULTS_DIR, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nFull metrics saved → {json_path}")

        # Save flat CSV summary (one row per algorithm)
        csv_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
        fieldnames = [
            "algorithm",
            # Efficiency
            "total_path_length",
            "avg_path_length",
            "makespan_steps",
            "avg_travel_time_steps",
            # Safety
            "collision_count",
            "near_miss_count",
            "min_separation_distance",
            "safety_violation_rate",
            # Computational
            "max_planning_time_s",
            "std_planning_time_s",
            "total_runtime_s",
            # Congestion
            "avg_active_agents_per_step",
            "avg_wait_time_steps",
            "total_wait_steps",
            # Fairness
            "travel_time_variance",
        ]

        rows = []
        for algo_name, m in all_metrics.items():
            row = {
                "algorithm": algo_name,
                **{k: v for k, v in m["efficiency"].items() if k != "path_length_per_plane"},
                **{k: v for k, v in m["safety"].items()},
                **{k: v for k, v in m["computational"].items()},
                **{k: v for k, v in m["congestion"].items()},
                "travel_time_variance": m["fairness"]["travel_time_variance"],
            }
            rows.append(row)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Metrics summary CSV saved → {csv_path}")