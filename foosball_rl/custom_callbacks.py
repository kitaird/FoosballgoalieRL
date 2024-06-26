from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    Source: stable-baselines3
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"][0]
        ball_position = infos["ball_position"]
        self.logger.record("custom/ball_position_x", ball_position[0])
        self.logger.record("custom/ball_position_y", ball_position[1])
        self.logger.record("custom/ball_position_z", ball_position[2])
        return True
