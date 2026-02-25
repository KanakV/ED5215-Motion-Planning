from atc_grid_env_central import Scenario, AlgoSimulation, MultiAlgorithmVisualizer
from planner import centralized_planner
from planner import centralized_planner
from planner import centralized_planner

scenario = Scenario(
    grid_size=50,
    max_planes=8,
    sim_time=60
)

simA = AlgoSimulation(scenario, centralized_planner, "Algorithm A")
simB = AlgoSimulation(scenario, centralized_planner, "Algorithm B")
simC = AlgoSimulation(scenario, centralized_planner, "Algorithm C")

visualizer = MultiAlgorithmVisualizer(
    scenario,
    [simA, simB, simC]
)

visualizer.run()