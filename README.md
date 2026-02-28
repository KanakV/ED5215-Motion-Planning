# ED5215 — Air Traffic Control Motion Planning

A multi-algorithm air traffic control (ATC) simulation and benchmarking framework built in Python. The project simulates multiple aircraft navigating a 2D grid toward a shared runway, running up to three planning algorithms **side-by-side on the same scenario** so their performance can be directly compared.

---

## What It Does

- Generates a random ATC scenario: a shared runway and a timed sequence of aircraft that spawn from the grid border
- Runs the same scenario simultaneously through up to **three different planning algorithms**
- Animates all three simulations in a live **side-by-side matplotlib visualization**, showing each plane's trail, its orientation (rotated plane icon), and proximity/collision warnings
- Exports a per-plane **path length comparison table** to `algorithm_comparison_table.csv` when the window is closed

---

## Project Structure

```
ED5215-Motion-Planning/
├── main.py                         # Entry point — wires scenario, simulations, and visualizer
├── planner.py                      # Planning algorithms (centralized_planner, simple_planner, ...)
├── map/
│   └── atc_grid_env_central.py     # Scenario, AlgoSimulation, MultiAlgorithmVisualizer classes
├── config/
│   └── config.py                   # Constants: GRID_SIZE, MAX_PLANES, SIM_TIME, PLANE_RADIUS, COLLISION_RADIUS
├── resources/
│   ├── plane.png                   # Plane icon (rotated per heading in animation)
│   ├── runway.png                  # Runway icon
│   └── background.jpg              # Grid background image
└── algorithm_comparison_table.csv  # Auto-generated results after each run
```

---

## How It Works

### Scenario
A `Scenario` is generated once and shared across all algorithm simulations to ensure a fair comparison. It contains:
- A **randomly placed runway** somewhere in the centre quarter of the grid
- A list of **spawn events** — `(time, border_position)` pairs that determine when and where each aircraft enters the airspace

### AlgoSimulation
Each `AlgoSimulation` wraps a planner function and runs against the shared scenario. On every simulation step it:
1. Spawns any aircraft whose scheduled time has arrived
2. Calls the **planner** with the current grid state, all active planes, and the runway position
3. Moves each plane to the position returned by the planner
4. Removes planes that have reached the runway
5. Tracks path length and compute time for each plane

### Planner Interface
A planner is any function with this signature:

```python
def my_planner(grid: np.ndarray, active_planes: list[dict], goal: tuple) -> dict:
    ...
    return actions  # {plane_id: (next_x, next_y)}
```

Each plane dict contains:
| Key | Description |
|---|---|
| `id` | Unique integer ID |
| `pos` | Current `(x, y)` position |
| `prev_pos` | Position from the previous step |
| `history` | Full list of past positions |
| `active` | `True` until the plane reaches the runway |
| `path_length` | Cumulative Euclidean distance travelled |

### Visualizer
`MultiAlgorithmVisualizer` runs a `matplotlib.animation.FuncAnimation` loop that steps all simulations forward in real time and renders them in a 1×3 panel layout. Visual features include:
- Plane icons **rotated to face their direction of travel**
- Coloured **path trails** per plane
- **Hollow red circles** when two planes enter a proximity warning zone
- **Filled red circles** when two planes are within the collision threshold
- A live **elapsed time** counter per panel

---

## Current Planners (`planner.py`)

### `simple_planner(grid, start, goal)`
A single-agent greedy planner. Moves one plane one step at a time along whichever axis reduces distance to the goal — no obstacle or conflict awareness. Used as a baseline.

### `centralized_planner(grid, active_planes, goal)`
A conflict-aware greedy planner that handles all active aircraft in a single call. On each step it:
- Sorts planes by ID (lower ID = higher priority)
- Generates the four cardinal neighbours + wait-in-place as candidate moves
- Filters out moves that go outside the grid bounds
- Sorts candidates by Euclidean distance to the runway
- Skips any cell already **reserved** by a higher-priority plane in this step
- Detects and skips **head-on swaps** (two planes trying to pass through each other)
- Assigns the best remaining candidate and reserves that cell

This is a **greedy, priority-based, single-step look-ahead** planner — no global path pre-computation.

---

## Configuration (`config/config.py`)

| Parameter | Description |
|---|---|
| `GRID_SIZE` | Width/height of the 2D airspace grid |
| `MAX_PLANES` | Maximum number of aircraft per scenario |
| `SIM_TIME` | Total simulated time (seconds) over which planes spawn |
| `PLANE_RADIUS` | Display radius of the plane icon and collision geometry |
| `COLLISION_RADIUS` | Distance threshold for a hard collision event |

---

## Getting Started

### Requirements
- Python 3.8+
- `numpy`
- `matplotlib`

```bash
pip install numpy matplotlib
```

### Run

```bash
python main.py
```

The animation window opens showing all three simulations running in parallel. Close the window to end the run — results are then written to `algorithm_comparison_table.csv`.

---

## Adding a New Algorithm

1. Write a planner function in `planner.py` following the interface above.
2. Import it in `main.py`.
3. Assign it to one of the simulation slots:

```python
from planner import centralized_planner, my_new_planner

simA = AlgoSimulation(scenario, centralized_planner, "Centralized Greedy")
simB = AlgoSimulation(scenario, my_new_planner,      "My New Planner")
simC = AlgoSimulation(scenario, centralized_planner,  "Algorithm C")

visualizer = MultiAlgorithmVisualizer(scenario, [simA, simB, simC])
visualizer.run()
```

> The visualizer currently expects exactly **three** simulations.

---

## Output

After the animation closes, `algorithm_comparison_table.csv` is written with the following structure:

| Plane | Algorithm A | Algorithm B | Algorithm C |
|---|---|---|---|
| 1 | 42.3 | 38.1 | 45.0 |
| 2 | 31.7 | 29.4 | 33.2 |
| … | … | … | … |
| avg | 37.0 | 33.8 | 39.1 |

Path lengths are cumulative Euclidean distances in grid units. Average compute time per planning call is also printed to the console.

---

## Course

Developed for **ED5215 — Motion Planning**, exploring centralized multi-agent planning strategies in a simulated air traffic control environment.
