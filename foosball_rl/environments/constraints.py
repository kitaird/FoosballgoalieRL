import numpy as np

from foosball_rl.environments.constants import BALL_VELOCITY_THRESHOLD, MAX_GOAL_Z_BOUNDS, \
    ABS_MAX_GOAL_Y_SYMMETRIC_BOUND, BLACK_GOAL_X_BOUNDS, WHITE_GOAL_X_BOUNDS, ABS_MAX_TABLE_X, ABS_MAX_TABLE_Y, \
    PLAYERS_POSITIONS, FIGURE_X_REACH_INCREMENT


def ball_stopped(ball_velocities, threshold=BALL_VELOCITY_THRESHOLD) -> bool:
    return np.linalg.norm(ball_velocities, np.inf) < threshold


def ball_outside_player_space(ball_data, player: str) -> bool:
    player_pos = PLAYERS_POSITIONS[player]
    in_x_reach = player_pos[0] - FIGURE_X_REACH_INCREMENT < ball_data[0] < player_pos[0] + FIGURE_X_REACH_INCREMENT
    in_y_reach = -player_pos[1] < ball_data[1] < player_pos[1]
    return not in_x_reach or not in_y_reach


def ball_outside_table(ball_data) -> bool:
    return np.abs(ball_data[0]) > ABS_MAX_TABLE_X or np.abs(ball_data[1]) > ABS_MAX_TABLE_Y


def ball_in_goal_bounds(ball_pos) -> bool:
    # Ball in y- and z- bounds of any goal
    return np.abs(ball_pos[1]) < ABS_MAX_GOAL_Y_SYMMETRIC_BOUND and MAX_GOAL_Z_BOUNDS[0] < ball_pos[2] < MAX_GOAL_Z_BOUNDS[1]


def ball_in_black_goal_bounds(ball_pos) -> bool:
    return BLACK_GOAL_X_BOUNDS[1] < ball_pos[0] < BLACK_GOAL_X_BOUNDS[0] and ball_in_goal_bounds(ball_pos)


def ball_in_white_goal_bounds(ball_pos) -> bool:
    return WHITE_GOAL_X_BOUNDS[0] < ball_pos[0] < WHITE_GOAL_X_BOUNDS[1] and ball_in_goal_bounds(ball_pos)


def black_goal_scored(sensors, ball_pos) -> bool:
    return sensors("black_goal_sensor").data[0] > 0 or ball_in_black_goal_bounds(ball_pos)


def white_goal_scored(sensors, ball_pos) -> bool:
    return sensors("white_goal_sensor").data[0] > 0 or ball_in_white_goal_bounds(ball_pos)
