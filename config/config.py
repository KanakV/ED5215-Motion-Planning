SIM_TIME = 60
GRID_SIZE = 50
MAX_PLANES = 15

PLANE_RADIUS = 1
COLLISION_RADIUS = PLANE_RADIUS 
WARNING_RADIUS = 1.5 * PLANE_RADIUS

MAX_NODES = 600
# Possible future changes

RUNWAY_START = (GRID_SIZE // 2, GRID_SIZE - 1)
RUNWAY_END = (GRID_SIZE // 2, GRID_SIZE - 5)
SPAWN_FREQUENCUY = 1

# LRA Config
LOOKAHEAD = 8              # slightly larger horizon
WAIT_ALLOWED = True

GOAL_WEIGHT = 1.2          # bias toward runway
CONFLICT_PENALTY = 3       # soft penalty instead of hard block
INERTIA_BONUS = 0.3        # reward continuing direction
