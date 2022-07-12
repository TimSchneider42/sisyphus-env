from typing import Dict, Tuple

import numpy as np

from robot_gym.core.sensors import ContinuousSensor
from transformation import Transformation
from .sisyphus_task import SisyphusTask


class FingertipPose2DSensor(ContinuousSensor[SisyphusTask]):
    def __init__(self, robot_name: str = "ur10", rotation_range: float = 0.3):
        assert 0 <= rotation_range <= np.pi / 4
        super(FingertipPose2DSensor, self).__init__()
        self.__robot_name = robot_name
        self.__rotation_range = rotation_range

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        lim: np.ndarray = self.task.table_top_finger_limits
        if self.__rotation_range > 0.0:
            return {
                "gripper_pos_xy": (-lim, lim),
                "gripper_angle_z": (-np.array([self.__rotation_range]), np.array([self.__rotation_range]))
            }
        else:
            return {
                "gripper_pos_xy": (-lim, lim)
            }

    def __observe(self) -> Dict[str, np.ndarray]:
        fingertip_pose = self.task.environment.robots[self.__robot_name].gripper.pose.transform(
            self.task.fingertip_pose_gripper_frame)
        fingertip_rotation = Transformation.from_pos_euler(euler_angles=np.array([0.0, np.pi, 0]), sequence="XYZ")
        rotated_fingertip_pose = self.task.fingertip_pose_gripper_frame * fingertip_rotation
        fingertip_pose_table_frame = self.task.table_top_center_pose.transform(
            fingertip_pose * rotated_fingertip_pose, inverse=True)
        angle = fingertip_pose_table_frame.rotation.as_euler("xyz")[2]
        pos_xy = fingertip_pose_table_frame.translation[:2]
        if self.__rotation_range > 0.0:
            return {
                "gripper_pos_xy": pos_xy,
                "gripper_angle_z": angle
            }
        else:
            return {
                "gripper_pos_xy": pos_xy
            }

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()
