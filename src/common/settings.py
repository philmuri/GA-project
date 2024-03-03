# Constants: General
WIDTH = 800
HEIGHT = 600
BASE_HEIGHT = 50
OBSTACLE_WIDTH = 200
OBSTACLE_HEIGHT_MIN = 200
OBSTACLE_HEIGHT_MAX = 300
BG_COLOR = (200, 200, 200)
BASE_COLOR = (0, 0, 0)
OBSTACLE_COLOR = (0, 0, 0)
GATE_CLOSED_COLOR = (60, 60, 60)
KEY_COLOR = (0, 255, 0)
KEY_SIZE = 50

OBSTACLE_SPEED = 5
GRAVITY = 0.5
JUMP_FORCE = -10
GAME_FPS = 90

PLAYER_START_POS = 70
PLAYER_START_HEIGHT = HEIGHT // 2
PLAYER_RADIUS = 20
# tied to GAME_FPS; actual cd is = PLAYER_JUMP_COOLDOWN when GAME_FPS = 60
PLAYER_JUMP_COOLDOWN = 0.5
PLAYER_COLOR = (128, 128, 128)
PLAYER_DEATH_COLOR = (255, 0, 0)

FONT_COLOR = (128, 128, 128)
FONT_INFO_COLOR = (255, 0, 0)
FONT_SIZE = 16
FONT_TYPE = 'Calibri'

# Constants: AI
is_AI = True
MAX_GENERATIONS = 100
POPULATION_SIZE = 50
MUTATION_CHANCE = 0.5  # mutation probability per weight
MUTATION_SIZE = 0.5  # value of 1 gives up to +-0.25 to weights
KEEP_PARENTS = 2
# The sum of the following must be below 1
# The remainder 1 - CROSSOVER_RATE - CROSS_GENERATION_RATE is for cloning and culling
CROSSOVER_RATE = 0.3
CROSS_GENERATION_RATE = 0.3
# culling: number of generations before partly randomizing genes if no performance improvement since
RESET_THRESHOLD = 10
# threshold in range [0,1] to pass prediction for player to jump. determined empirically to achieve a
# balance between cautiosness and aggressiveness in jumping
DECISION_THRESHOLD = 0.5
LOSS_PENALTY = 1  # death/loss scaling relative to score points
# Evaluation/performance metrics:
EM_KSUCCESS = 5  # last k generations for jump success rate evaluation
