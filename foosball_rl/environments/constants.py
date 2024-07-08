"""
    Dimensions of the playing field:
    - Long side of the kicker (x-axis) : [-0.67, 0.67]
    - Short side of the kicker (y-axis): [-0.355, 0.355]
    - Height of the kicker (z-axis): [0.0, 0.9]
"""

ABS_MAX_TABLE_X = 0.67
ABS_MAX_TABLE_Y = 0.355
MAX_TABLE_Z_BOUNDS = [0.0, 0.9]

BLACK_GOAL_X_BOUNDS = [-0.608, -0.688]
WHITE_GOAL_X_BOUNDS = [-x for x in BLACK_GOAL_X_BOUNDS]

ABS_GOAL_Y_SYMMETRIC_BOUND = 0.0875
ABS_MAX_GOAL_Y_SYMMETRIC_BOUND = 0.15  # Considers the width of the area behind a goal
GOAL_Z_BOUNDS = [0.305, 0.3705]
MAX_GOAL_Z_BOUNDS = [-0.1705, 0.3705]  # Considers the height of the area behind a goal

BLACK_GOAL_X_POSITION = -0.625
WHITE_GOAL_X_POSITION = -BLACK_GOAL_X_POSITION

FIELD_HEIGHT = 0.3075  # Setting the z-coordinate of the ball to this value will put it on the field

FIGURE_RADIUS = 0.072  # The radius of the figure of a player
FIGURE_X_REACH_INCREMENT = 0.06595  # The largest x-coordinate-increment a player can reach in both directions
# Calculated based on the diameter of the ball (d=0.035), the diameter of the figure (D=0.144), the height of the figure (b=0.006) and the euclidean distance

GOALKEEPER_Y_ABS_REACH = 0.1225
DEFENCE_Y_ABS_REACH = 0.182
MIDFIELD_Y_ABS_REACH = 0.06
STRIKER_Y_ABS_REACH = 0.1225

PLAYER_X_RANGE_INCREMENT = 0.14915  # The largest x-coordinate-increment a player can reach
PLAYER_BALL_DISTANCE_INCREMENT = 0.0375  # Setting the x-coordinate of the ball to this value + any X position of a player will put the ball to the right of the player (with the perspective of the black goal being on the left side)

BALL_VELOCITY_THRESHOLD = 0.01  # Threshold for the ball to be considered stopped

# Positions

BLACK_GOALKEEPER_X_POSITION = -0.522
BLACK_DEFENCE_X_POSITION = -0.3728
BLACK_MIDFIELD_X_POSITION = -0.075
BLACK_STRIKER_X_POSITION = 0.22395

WHITE_GOALKEEPER_X_POSITION = -BLACK_GOALKEEPER_X_POSITION
WHITE_DEFENCE_X_POSITION = -BLACK_DEFENCE_X_POSITION
WHITE_MIDFIELD_X_POSITION = -BLACK_MIDFIELD_X_POSITION
WHITE_STRIKER_X_POSITION = -BLACK_STRIKER_X_POSITION

PLAYERS_POSITIONS = {
    "b_g": [BLACK_GOALKEEPER_X_POSITION, GOALKEEPER_Y_ABS_REACH],
    "b_d": [BLACK_DEFENCE_X_POSITION, DEFENCE_Y_ABS_REACH],
    "b_m": [BLACK_MIDFIELD_X_POSITION, MIDFIELD_Y_ABS_REACH],
    "b_s": [BLACK_STRIKER_X_POSITION, STRIKER_Y_ABS_REACH],
    "w_g": [WHITE_GOALKEEPER_X_POSITION, GOALKEEPER_Y_ABS_REACH],
    "w_d": [WHITE_DEFENCE_X_POSITION, DEFENCE_Y_ABS_REACH],
    "w_m": [WHITE_MIDFIELD_X_POSITION, MIDFIELD_Y_ABS_REACH],
    "w_s": [WHITE_STRIKER_X_POSITION, STRIKER_Y_ABS_REACH]
}
