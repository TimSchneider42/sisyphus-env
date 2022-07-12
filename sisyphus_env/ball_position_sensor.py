from abc import ABC
from typing import Dict, Tuple

import numpy as np

from robot_gym.core.sensors import ContinuousSensor
from .sisyphus_task import SisyphusTask


class BallPositionSensor(ContinuousSensor[SisyphusTask], ABC):
    def __init__(self):
        super(BallPositionSensor, self).__init__(clip=False)

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        te = self.task.table_top_accessible_extents
        return {"ball_position": (-te / 2, te / 2)}

    def __observe(self) -> Dict[str, np.ndarray]:
        return {"ball_position": self.task.ball_2d_position_table_frame}

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()
