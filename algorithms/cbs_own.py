from dataclasses import dataclass,field
from typing import Tuple, Dict, List

AgentId = int
Vertex = Tuple[int, int]
Time = int

@dataclass
class Constraint:
    agent: AgentId
    vertex: Vertex
    time: Time

@dataclass
class Conflict:
    agent_i: AgentId
    agent_j: AgentId
    vertex: Vertex
    time: Time

# Allows dataclass to be compared (by costs)
@dataclass(order=True)
class CTNode:
    constraints: Dict[AgentId, List[Constraint]] = field(compare=False)
    solution: Dict[AgentId, List[Vertex]] = field(compare=False)
    cost: int

@dataclass
class EdgeConflcit:
    agent_i: AgentId
    agent_j: AgentId
    vertex_A: Vertex
    vertex_B: Vertex
    time: Time


def AStarConstrained(
        agent: AgentId,
        start: Vertex,
        goal: Vertex,
        constraints: List[Constraint]
):
    ...
