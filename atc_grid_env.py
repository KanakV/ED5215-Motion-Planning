import numpy as np
import random
import math
import time
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D
from matplotlib.patches import Circle


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
# SINGLE ALGORITHM SIMULATION
# =====================================================

class AlgoSimulation:

    def __init__(self, scenario, planner, name):

        self.name = name
        self.planner = planner

        self.grid_size = scenario.grid_size
        self.runway = scenario.runway
        self.spawn_events = scenario.spawn_events.copy()

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self.plane_radius = 2.2
        self.collision_threshold = self.plane_radius * 2

        self.planes = []
        self.plane_counter = 0

        self.total_compute_time = 0
        self.planning_calls = 0

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
                "color": np.random.rand(3,)
            }

            self.plane_counter += 1
            self.planes.append(plane)

        # Real-time replanning
        for plane in self.planes:

            if not plane["active"]:
                continue

            start = time.time()

            path = self.planner(
                self.grid.copy(),
                plane["pos"],
                self.runway
            )

            compute = time.time() - start
            self.total_compute_time += compute
            self.planning_calls += 1

            if path:

                next_pos = path[0]

                plane["prev_pos"] = plane["pos"]

                plane["path_length"] += np.linalg.norm(
                    np.array(next_pos) - np.array(plane["pos"])
                )

                plane["pos"] = next_pos
                plane["history"].append(next_pos)

            if plane["pos"] == self.runway:
                plane["active"] = False

        self._update_grid()

    # -------------------------------------------------

    def _update_grid(self):
        self.grid[:] = 0
        self.grid[self.runway] = 1

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


# =====================================================
# MULTI PANEL VISUALIZER
# =====================================================

class MultiAlgorithmVisualizer:

    def __init__(self, scenario, simulations):

        self.scenario = scenario
        self.simulations = simulations

        # Add left/right margins
        self.fig, self.axes = plt.subplots(1, 3, figsize=(20, 6))
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, wspace=0.15)

        self.plane_img = plt.imread("plane.png")
        self.runway_img = plt.imread("runway.png")
        self.background_img = plt.imread("background.jpg")

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
        self._export_results()

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

            # Collision visualization
            for i in range(len(sim.planes)):
                for j in range(i + 1, len(sim.planes)):

                    p1 = sim.planes[i]
                    p2 = sim.planes[j]

                    if not p1["active"] or not p2["active"]:
                        continue

                    dist = np.linalg.norm(
                        np.array(p1["pos"]) - np.array(p2["pos"])
                    )

                    if dist < sim.collision_threshold:
                        ax.add_patch(
                            Circle(p1["pos"],
                                   sim.plane_radius * 1.2,
                                   fill=False,
                                   edgecolor="red",
                                   linewidth=2,
                                   zorder=4)
                        )
                        ax.add_patch(
                            Circle(p2["pos"],
                                   sim.plane_radius * 1.2,
                                   fill=False,
                                   edgecolor="red",
                                   linewidth=2,
                                   zorder=4)
                        )

            ax.set_title(sim.name)

            # Timer
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

    def _export_results(self):

        print("\n==== FINAL RESULTS ====\n")

        # Collect per-plane data
        algo_names = [sim.name for sim in self.simulations]
        plane_count = max(len(sim.planes) for sim in self.simulations)

        # Build table dictionary
        table = []

        for i in range(plane_count):

            row = {"Plane": i + 1}

            for sim in self.simulations:

                if i < len(sim.planes):
                    row[sim.name] = sim.planes[i]["path_length"]
                else:
                    row[sim.name] = ""

            table.append(row)

        # Compute averages
        avg_row = {"Plane": "avg"}

        for sim in self.simulations:

            total = sum(p["path_length"] for p in sim.planes)
            avg = total / len(sim.planes) if sim.planes else 0
            avg_row[sim.name] = avg

            print(f"{sim.name} Average Path Length: {avg:.3f}")

        table.append(avg_row)

        # Write CSV
        fieldnames = ["Plane"] + algo_names

        with open("algorithm_comparison_table.csv", "w", newline="") as f:

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(table)

        print("\nComparison table saved to algorithm_comparison_table.csv")