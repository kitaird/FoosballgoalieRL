import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs


class VecPBRSWrapper(VecEnvWrapper):

    def __init__(self, venv: VecEnv, gamma: float = 0.99):
        VecEnvWrapper.__init__(self, venv, venv.observation_space, venv.action_space)
        self.venv = venv
        self.gamma = gamma
        self.last_potentials = None

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.last_potentials = np.zeros(self.num_envs)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        current_potentials = _potentials(infos)
        potential_differences = self.gamma * current_potentials - self.last_potentials
        self.last_potentials = current_potentials
        rewards += potential_differences
        return observations, rewards, dones, infos


def _potential(ball_pos: np.ndarray):
    return weighted_x_pot(ball_pos[0]) + weighted_z_pot(ball_pos[1])


def _potentials(infos: list[dict]):
    return np.array([_potential(info["ball_position"]) for info in infos])


def x_pot(x):
    # max x -1.217 , 1.217
    val = x * 1 / 2.42 + 0.5
    return np.clip(val, 0, 1)


def weighted_x_pot(x):
    return x_pot(x) * 4 / 5


def z_pot(z):
    # max z -0.68 , 0.68
    if z < 0:
        pot = z * 2.08 + 1.42
    else:
        pot = z * -2.08 + 1.42
    return np.clip(pot, 0, 1)


def weighted_z_pot(z):
    return z_pot(z) / 5


def contour_plot_to_console(X, Z, Y):
    for i in range(len(X)):
        if i % 5 == 0:
            row = ""
            for j in range(len(Z)):
                if j % 5 == 0:
                    row += " " + f"{Y[i][j]:.2f}"
            print(row)
