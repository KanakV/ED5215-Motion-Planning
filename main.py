from map.atc_grid_env_central import Scenario, AlgoSimulation, MultiAlgorithmVisualizer
from planner import centralized_planner
from planner import centralized_planner
from planner import centralized_planner

from config.config import SIM_TIME, GRID_SIZE, MAX_PLANES


# Maybe add runway position?
# Creates the Map
# Has runway and spawn points with time at which the planes spawn
scenario = Scenario(
    grid_size=GRID_SIZE,
    max_planes=MAX_PLANES,
    sim_time=SIM_TIME
)

simA = AlgoSimulation(scenario, centralized_planner, "Algorithm A")
simB = AlgoSimulation(scenario, centralized_planner, "Algorithm B")
simC = AlgoSimulation(scenario, centralized_planner, "Algorithm C")

visualizer = MultiAlgorithmVisualizer(
    scenario,
    [simA, simB, simC]
)

visualizer.run()