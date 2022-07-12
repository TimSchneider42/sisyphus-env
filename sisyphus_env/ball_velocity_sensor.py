from abc import ABC
from typing import Dict, Tuple

import numpy as np

from robot_gym.core.sensors import ContinuousSensor
from .sisyphus_task import SisyphusTask


class BallVelocitySensor(ContinuousSensor[SisyphusTask], ABC):
    def __init__(self, vmax: float = 0.2):
        super(BallVelocitySensor, self).__init__(clip=False)
        self.__vmax = vmax

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {"ball_velocity": (np.array([-self.__vmax, -self.__vmax]), np.array([self.__vmax, self.__vmax]))}

    def __observe(self) -> Dict[str, np.ndarray]:
        return {"ball_velocity": self.task.ball_2d_velocity_table_frame}

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()
