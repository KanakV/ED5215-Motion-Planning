from environment.atc_grid_env_central import Scenario, AlgoSimulation, MultiAlgorithmVisualizer
from config.config import SIM_TIME, GRID_SIZE, MAX_PLANES

# Import Algorithms
from algos.cooperative_astar import cooperative_planner
from algos.longrange_astar import lra_planner
from algos.cbs_point import cbs_planner

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