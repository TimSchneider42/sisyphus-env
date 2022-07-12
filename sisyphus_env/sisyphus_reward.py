from typing import Tuple

import numpy as np

from robot_gym.core.rewards import NormalizedReward
from .sisyphus_task import SisyphusTask


class SisyphusReward(NormalizedReward[SisyphusTask]):
    def __init__(self, dense: bool = False):
        super(SisyphusReward, self).__init__(name="ball_placing_reward")
        self.__dense = dense

    def _calculate_reward_unnormalized(self) -> float:
        ball_pos_target_frame = self.task.ball_2d_position_table_frame - self.task.target_zone_pos_table_frame
        if self.__dense:
            return 1.0 - min(float(np.linalg.norm(ball_pos_target_frame)), 1.0)
        else:
            ball_in_target_zone = np.all(np.logical_and(ball_pos_target_frame <= self.task.target_zone_extents / 2,
                                                        ball_pos_target_frame >= -self.task.target_zone_extents / 2))
            return 1.0 if ball_in_target_zone else 0.0

    def _get_reward_bounds(self) -> Tuple[float, float]:
        return 0, 1
