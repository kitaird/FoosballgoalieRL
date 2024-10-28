import functools

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID, ObsType, ActionType

from foosball_rl.environments.foosball.single_agent.foosball_rl import RawEnv


class FoosballMARL(ParallelEnv):
    def __init__(self, env: RawEnv,
                 use_team_agents: bool = True):
        super().__init__()
        ParallelEnv.metadata = env.metadata
        self.env: RawEnv = env
        if use_team_agents:
            possible_agents = ["black_team", "white_team"]
        else:
            possible_agents = [f"{team}_{agent}" for agent in ["goalkeeper", "defense", "midfield", "striker"] for team
                               in ["black", "white"]]
        self.possible_agents: list[AgentID] = possible_agents
        self.agents: list[AgentID] = possible_agents
        self.observation_spaces = dict(zip(self.agents, [self.env.observation_space] * len(self.agents)))

        low, high = np.split(self.env.mj_model.actuator_ctrlrange.copy().astype(np.float32), 2 if use_team_agents else 8)[0].T
        self._marl_action_space: gym.spaces.Space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[
        dict[AgentID, ObsType], dict[AgentID, dict]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            for i, agent in enumerate(self.agents):
                self.observation_space(agent).seed(seed)
                self.action_space(agent).seed(seed)

        return {agent: obs for agent in self.agents}, {agent: info for agent in self.agents}

    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType], dict[AgentID, float], dict[AgentID, bool], dict[AgentID, bool], dict[AgentID, dict]]:

        obs, rew, terminated, truncated, info = self.env.step(actions)

        all_obs = {agent: obs for agent in self.agents}
        all_rewards = {agent: rew if agent.startswith("black") else -rew for agent in self.agents}
        all_term = {agent: terminated for agent in self.agents}
        all_trunc = {agent: truncated for agent in self.agents}
        all_infos = {agent: info for agent in self.agents}
        return all_obs, all_rewards, all_term, all_trunc, all_infos

    def render(self) -> None | np.ndarray | str | list:
        return self.env.render()

    def close(self):
        self.env.close()

    def state(self) -> np.ndarray:
        return self.env.get_observation()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gym.spaces.Space:
        return self.env.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gym.spaces.Space:
        return self._marl_action_space
