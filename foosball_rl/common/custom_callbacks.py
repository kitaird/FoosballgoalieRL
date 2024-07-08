from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    Based on https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        ################################
        # Add custom tensorboard values here, e.g.:
        # infos = self.locals["infos"][0]
        # self.logger.record("custom/ball_position_x", infos["ball_x_position"])
        ################################
        return True
