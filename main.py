from map.atc_grid_env_central import Scenario, AlgoSimulation, MultiAlgorithmVisualizer
from planner2 import cooperative_planner
from planner3 import lra_planner
from planner4 import cbs_planner

from config.config import SIM_TIME, GRID_SIZE, MAX_PLANES


# Maybe add runway position?
# Creates the Map
# Has runway and spawn points with time at which the planes spawn
scenario = Scenario(
    grid_size=GRID_SIZE,
    max_planes=MAX_PLANES,
    sim_time=SIM_TIME
)

simA = AlgoSimulation(scenario, cooperative_planner, "Algorithm A")
simB = AlgoSimulation(scenario, lra_planner, "Algorithm B")
#simC = AlgoSimulation(scenario, cbs_planner, "Algorithm C")

visualizer = MultiAlgorithmVisualizer(
    scenario,
    [simA, simB]
)

visualizer.run()