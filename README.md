# Training a Goalie using Deep Reinforcement Learning

This is an exercise for students to train a goalie in a foosball environment using Python.
It includes the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and [stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) repositories, which provide various
reinforcement learning algorithms.
Students are encouraged to experiment with different algorithms and observe their performance.
The results of multiple runs are aggregated using the
provided [tensorboard-aggregator](https://github.com/philipperemy/tensorboard-aggregator) repository.

## Installing

Navigate to the project directory and install the required Python packages.

```bash
pip install -r requirements.txt
```

## Running the Project

The main entry point of the project is `src/__main__.py`. This script reads the configuration
from `resources/config.ini`, creates the environment, and starts the training process.
One run will train 5 different models with the same configuration but different random-seeds and save the results in
the `experiments` directory.

#### Before training: Make sure to set the `experiment_name` in the `resources/config.ini` file to a unique name for the current experiment.

```bash
python src/__main__.py
```

After the training process is finished, the aggregated results can be viewed using tensorboard.

```bash
tensorboard --logdir ./experiments/{experiment_name}/tensorboard/aggregates
```

Make sure to replace `{experiment_name}` with the name of the experiment you want to view.

## Configuration

The project's configuration is stored in `resources/config.ini`. This file contains several sections, each with its own
set of parameters.

- `Common`: General experiment settings, such as the experiment name where to store a copy of the used
  run-configuration.
- `Kicker`: Settings for the kicker environment.
- `Algorithm`: Settings for the DRL-algorithm.
- `Training`: Settings for the training process.

The following sections are not necessary for the training process but provide additional functionality:

- `Callback`: Settings for callbacks during training. E.g. where to store intermediate models and how often to save them.
- `VideoRecording`: Settings for recording videos of the training process. These specify where to store the videos and
  how often to record them.

## The Goal <p style="font-size:12px">*no pun intended*</p>

The goal is to train a goalie that blocks the ball from entering the goal and is able to score goals itself.
Given the provided training steps, the students should experiment with different algorithms and hyperparameters to
achieve the best possible performance.
Additionally, students should work with different environment wrappers provided by stable-baselines3 to improve the
training process.

During training several snapshots of the model are saved in the `experiments` directory.
The maximum training length is 2e6 steps.

## MDP Environment

The Markov Decision Process (MDP) for this project is defined in the `src/kicker/kicker_env.py` file. It represents the
foosball environment where the goalie is trained.

### Observation Space

The observation space represents the current state of the foosball environment. It includes sensor readings from the
ball and the goalie.

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
| Goalie - last action - lateral |   [-1, 1]   | float |
| Goalie - last action - angular |   [-1, 1]   | float |

### Action Space

The action space represents the possible actions the goalie can take. It includes moving in a lateral or angular
direction.
The action space is continuous if `continuous_act_space` is set to `True` in the `config.ini` file, and
multi-discrete if `multi_discrete_act_space` is set to `True`. 
If both are set to `False`, the action space is one dimensional discrete.
The binning of the action space can be changed in the `config.ini`: `lateral_binning` 
and `angular_binning`.

The step frequency can be changed in the `config.ini` file by setting the `step_frequency` parameter. 
This parameter defines how many steps the action is repeated before a new action is taken (naturally, also influencing 
the observation frequency).

The action space is defined as follows:

#### Continuous Action Space

| Action channel            |  Range  | Type  |
|:--------------------------|:-------:|:-----:|
| Goalie - lateral - torque | [-1, 1] | float |
| Goalie - angular - torque | [-1, 1] | float |

#### Multi-Discrete Action Space

| Action channel            |    Range    | Type |
|:--------------------------|:-----------:|:----:|
| Goalie - lateral - torque | Discrete(5) | int  |
| Goalie - angular - torque | Discrete(5) | int  |

#### Discrete Action Space

| Action channel                       |    Range     | Type |
|:-------------------------------------|:------------:|:----:|
| Goalie - lateral or angular - torque | Discrete(10) | int  |



### Reward Function

A reward of `1` is given to the goalie for scoring a goal and a reward of `-1` is given to the goalie for conceding goal.

### Episode definition

An episode ends when:
- the ball enters any goal, 
- the `horizon` number of steps, as specified in the `config.ini` file, is reached
- the ball is out of bounds
- the ball stops outside the reach of the goalie

The environment is then reset for the next episode:
- The ball is placed at a random position on the opponent's side striker rod line
- The goalie is placed at the center of the goal
- The ball is given a random velocity shooting at the goalie