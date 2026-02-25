from atc_grid_env import Scenario, AlgoSimulation, MultiAlgorithmVisualizer
from planner import simple_planner
from planner import simple_planner
from planner import simple_planner

scenario = Scenario(
    grid_size=100,
    max_planes=8,
    sim_time=60
)

simA = AlgoSimulation(scenario, simple_planner, "Algorithm A")
simB = AlgoSimulation(scenario, simple_planner, "Algorithm B")
simC = AlgoSimulation(scenario, simple_planner, "Algorithm C")

visualizer = MultiAlgorithmVisualizer(
    scenario,
    [simA, simB, simC]
)

visualizer.run()