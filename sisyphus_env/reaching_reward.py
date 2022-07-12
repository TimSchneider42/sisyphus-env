from typing import Tuple

import numpy as np

from robot_gym.core.rewards import NormalizedReward
from .sisyphus_task import SisyphusTask


class ReachingReward(NormalizedReward[SisyphusTask]):
    def __init__(self, dense: bool = False):
        super(ReachingReward, self).__init__(name="reaching_reward")
        self.__dense = dense

    def _calculate_reward_unnormalized(self) -> float:
        gripper_pose = self.task.environment.robots["ur10"].gripper.pose
        gripper_pose_table_frame = self.task.table_top_center_pose.transform(gripper_pose, inverse=True)
        finger_pos_target_frame = gripper_pose_table_frame.translation[:2] - self.task.target_zone_pos_table_frame
        if not self.__dense:
            finger_in_target_zone = np.all(np.logical_and(
                finger_pos_target_frame <= self.task.target_zone_extents / 2,
                finger_pos_target_frame >= -self.task.target_zone_extents / 2))
            return 1.0 if finger_in_target_zone else 0.0
        else:
            return max(0.0, 1.0 - np.linalg.norm(finger_pos_target_frame))

    def _get_reward_bounds(self) -> Tuple[float, float]:
        return 0, 1
