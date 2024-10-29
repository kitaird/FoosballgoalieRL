import numpy as np
from gymnasium.utils import seeding


class EpisodeDefinition:

    def __init__(self):
        self.mj_data = None
        self._np_random = None

    def initialize_episode(self):
        raise NotImplementedError

    def is_truncated(self) -> bool:
        raise NotImplementedError

    def is_terminated(self) -> bool:
        raise NotImplementedError

    def seed(self, seed=None) -> None:
        if seed is not None:
            self._np_random, _ = seeding.np_random(seed)

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    @property
    def mj_data(self):
        return self._mj_data

    @mj_data.setter
    def mj_data(self, mj_data):
        self._mj_data = mj_data
