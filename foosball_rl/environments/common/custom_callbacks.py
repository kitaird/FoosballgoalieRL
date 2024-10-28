import os
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    Based on https://stable-baselines3.readthedocs.io/en/v2.3.2/guide/tensorboard.html#logging-more-values
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        ################################
        # <<ExtensionPoint>>: You can add custom tensorboard values here, e.g.:
        # infos = self.locals["infos"][0]
        # self.logger.record("custom/ball_position_x", infos["ball_x_position"])
        ################################
        return True


class SaveVecNormalizeAndRolloutBufferCallback(BaseCallback):
    """
    Based on: https://rl-baselines3-zoo.readthedocs.io/en/v2.3.0/_modules/rl_zoo3/callbacks.html#SaveVecNormalizeCallback
    and https://stable-baselines3.readthedocs.io/en/v2.3.2/guide/callbacks.html#checkpointcallback

    Callback for saving a VecNormalize wrapper and Rollout-Buffer every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param save_replay_buffer: (bool) Whether to save the replay buffer or not
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self,
                 save_freq: int,
                 save_path: Path,
                 save_replay_buffer: bool = True,
                 name_prefix: Optional[str] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_replay_buffer = save_replay_buffer
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if not self.save_path.exists():
            self.save_path.mkdir()

    def _on_step(self) -> bool:
        # make mypy happy
        assert self.model is not None

        if self.n_calls % self.save_freq == 0:
            self.store_vec_normalize()
            if self.save_replay_buffer:
                self.store_replay_buffer()
        return True

    def store_vec_normalize(self):
        if self.model.get_vec_normalize_env() is not None:
            path = self._save_path("vecnormalize")
            self.model.get_vec_normalize_env().save(path)  # type: ignore[union-attr]
            if self.verbose > 1:
                print(f"Saving VecNormalize to {path}")

    def store_replay_buffer(self):
        if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            replay_buffer_path = self._save_path("replay_buffer")
            self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
            if self.verbose > 1:
                print(f"Saving model replay buffer to {replay_buffer_path}")

    def _save_path(self, file_name: str):
        if self.name_prefix is not None:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps_{file_name}.pkl")
        else:
            path = os.path.join(self.save_path, f"{file_name}.pkl")
        return path

