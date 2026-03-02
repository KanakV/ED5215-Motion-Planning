from environment.atc_grid_env_central import Scenario, AlgoSimulation, MultiAlgorithmVisualizer
# from planner2 import cooperative_planner
# from planner3 import lra_planner
# from planner4 import cbs_planner

from config.config import SIM_TIME, GRID_SIZE, MAX_PLANES

# from algorithms.cbs_claude import cbs_planner
# from algorithms.cbs_claude2 import cbs_planner

# Import Algorithms
from algos.cooperative_astar import cooperative_planner
from algos.longrange_astar import lra_planner
from algos.cbs import cbs_planner

# Maybe add runway position?
# Creates the Map
# Has runway and spawn points with time at which the planes spawn
scenario = Scenario(
    grid_size=GRID_SIZE,
    max_planes=MAX_PLANES,
    sim_time=SIM_TIME
)

simA = AlgoSimulation(scenario, cooperative_planner, "Cooperative")
simB = AlgoSimulation(scenario, lra_planner, "Long Range A*")
simC = AlgoSimulation(scenario, cbs_planner, "CBS")

visualizer = MultiAlgorithmVisualizer(
    scenario,
    [simA, simB, simC]
)

visualizer.run()