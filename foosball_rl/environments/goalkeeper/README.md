# Goalkeeper-v0

This environment contains a single goalkeeper that tries to prevent the ball from entering the goal. The goalkeeper can move
laterally and rotate to block the ball. The ball is placed at a random position on the opponent's side striker rod line and
is given a random velocity shooting at the goal. The goalkeeper is placed at the center of the goal.
The following section describes the configuration possibilities of this environment.

| Goalkeeper-v0                                                      |
|--------------------------------------------------------------------|
| ![Goalkeeper-v0 Environment](../../../example-images/Goalkeeper-v0.png) |

## Observation Space

The observation space represents the current state of the Goalkeeper-v0 environment. It includes accurate sensor readings 
from the ball and the goalie.

| Observation channel            |    Range    | Type  |
|:-------------------------------|:-----------:|:-----:|
| Ball - position - x            | [-inf, inf] | float |
| Ball - position - z            | [-inf, inf] | float |
| Ball - position - y            | [-inf, inf] | float |
| Ball - velocity - x            | [-inf, inf] | float |
| Ball - velocity - y            | [-inf, inf] | float |
| Ball - velocity - z            | [-inf, inf] | float |
| Ball - acceleration - x        | [-inf, inf] | float |
| Ball - acceleration - y        | [-inf, inf] | float |
| Ball - acceleration - z        | [-inf, inf] | float |
| Goalie - position - lateral    | [-inf, inf] | float |
| Goalie - velocity - lateral    | [-inf, inf] | float |
| Goalie - position - angular    | [-inf, inf] | float |
| Goalie - velocity - angular    | [-inf, inf] | float |

## Action Space

The action space represents the possible actions the goalie can take. It includes moving in a lateral or angular
direction.  The action space is continuous by nature. However, it can be discretized by using the `ActionSpaceWrapper` referenced 
in the `foosball_rl/run_config.yml`.

The step frequency can be changed in the `goalkeeper-config.yml` file by setting the `step_frequency` parameter. 
This parameter defines how many steps the action is repeated before a new action is taken (naturally, also influencing 
the observation frequency).

The action space is defined as follows:

### Continuous Action Space

| Action channel            |  Range  | Type  |
|:--------------------------|:-------:|:-----:|
| Goalie - lateral - torque | [-1, 1] | float |
| Goalie - angular - torque | [-1, 1] | float |

### Multi-Discrete Action Space

| Action channel            |          Range          | Type |
|:--------------------------|:-----------------------:|:----:|
| Goalie - lateral - torque | Discrete(lateral_bins)  | int  |
| Goalie - angular - torque | Discrete(angular_bins)  | int  |

### Discrete Action Space

| Action channel                       |            Range                           | Type |
|:-------------------------------------|:------------------------------------------:|:----:|
| Goalie - lateral or angular - torque |   Discrete(lateral_bins + angular_bins)    | int  |


## Reward Function

A reward of `1` is given to the goalie for scoring a goal and a reward of `-1` is given to the goalie for conceding goal.

## Episode definition
The environment configuration file `foosball_rl/environments/goalkeeper/goalkeeper-config.yml` contains an `Environment`- and an `EpisodeDefinition`-section. The `Environment`-section is described in the [Environments README file](https://github.com/kitaird/FoosballRL/blob/develop/foosball_rl/environments/README.md). The `EpisodeDefinition`-section contains the following parameters: 
 - `EpisodeDefinition`: 
    - `end_episode_on_struck_goal`: Whether to end the episode when the ball enters the opponent's goal.
    - `end_episode_on_conceded_goal`: Whether to end the episode when the ball enters the own goal.
    - `end_episode_on_ball_stopped`: Whether to end the episode when the ball stops outside the reach of the goalie.
    - `ball_stopped_time_threshold_in_seconds`: The time threshold in seconds for the ball to stop inside the reach of the goalie. If `null`, the episode will not end early if the ball is still in reach (the environment horizon will determine the episode length).
    - `reset_goalie_position_on_episode_start`: Whether to reset the goalie position to the center of the goal at the beginning of each episode.

The initial implementation is as follows:

An episode ends when:
- (terminated) the ball enters any goal, 
- (truncated) the `horizon` number of steps is reached
- (truncated) the ball is out of bounds
- (truncated) the ball stops outside the reach of the goalie
- (truncated) the ball stops for too long (4s)

The environment is then reset for the next episode:
- The ball is placed at a random position on the opponent's side striker rod line
- The goalie is placed at the center of the goal
- The ball is given a random velocity shooting at the goal
