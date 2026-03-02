import random

SIM_TIME = 60
GRID_SIZE = 50

# Changes
MAX_PLANES = 15
RUNWAY = (GRID_SIZE // 2, GRID_SIZE // 2)
# RUNWAY = (random.randint(GRID_SIZE // 4, 3* GRID_SIZE // 4), random.randint(GRID_SIZE // 4, 3* GRID_SIZE // 4))
SPAWN_FREQUENCUY = 1

# Plane characteristics
PLANE_RADIUS = 1
COLLISION_RADIUS = PLANE_RADIUS 
WARNING_RADIUS = 1.5 * PLANE_RADIUS

# CBS Config
MAX_NODES = 1000

# LRA Config
LOOKAHEAD = 8              
WAIT_ALLOWED = True

GOAL_WEIGHT = 1.2         
CONFLICT_PENALTY = 3       
INERTIA_BONUS = 0.3        

# SpaceTime A* Config
SPAWN_BIAS_STEPS = 8