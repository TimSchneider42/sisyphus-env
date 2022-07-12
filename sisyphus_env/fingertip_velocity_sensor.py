from typing import Dict, Tuple, List

import numpy as np

from robot_gym.core.sensors import ContinuousSensor
from .sisyphus_task import SisyphusTask


class FingertipVelocity2DSensor(ContinuousSensor[SisyphusTask]):
    def __init__(self, robot_name: str, linear_limit_lower: np.ndarray, linear_limit_upper: np.ndarray,
                 angular_limit_lower: float, angular_limit_upper: float, sense_angle: bool = True):
        super(FingertipVelocity2DSensor, self).__init__()
        self.__robot_name = robot_name
        self.__sense_angle = sense_angle
        self.__linear_limit_lower = linear_limit_lower
        self.__linear_limit_upper = linear_limit_upper
        self.__angular_limit_lower = angular_limit_lower
        self.__angular_limit_upper = angular_limit_upper

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        lin_lims = (self.__linear_limit_lower, self.__linear_limit_upper)
        ang_lims = (np.array([self.__angular_limit_lower]), np.array([self.__angular_limit_upper]))
        if self.__sense_angle:
            return {
                "gripper_vel_xy": lin_lims,
                "gripper_angular_vel_z": ang_lims
            }
        else:
            return {
                "gripper_vel_xy": lin_lims
            }

    def __observe(self) -> Dict[str, np.ndarray]:
        gripper = self.task.environment.robots[self.__robot_name].gripper
        table = self.task.table_top_center_pose
        tcp_vel_lin_world_frame, tcp_vel_ang_world_frame = gripper.velocity
        tcp_to_fingertip_translation_world_frame = gripper.pose.rotation.apply(
            self.task.fingertip_pose_gripper_frame.translation)
        fingertip_ang_world_frame = tcp_vel_ang_world_frame
        rotation_induced_linear_vel = np.cross(tcp_vel_ang_world_frame, tcp_to_fingertip_translation_world_frame)
        tcp_lin_table_frame = tcp_vel_lin_world_frame + rotation_induced_linear_vel
        fingertip_lin_table_frame, fingertip_ang_table_frame = table.rotation.apply(
            (tcp_lin_table_frame, fingertip_ang_world_frame), inverse=True)
        if self.__sense_angle:
            return {
                "gripper_vel_xy": fingertip_lin_table_frame[:2],
                "gripper_angular_vel_z": fingertip_ang_table_frame[2:3]
            }
        else:
            return {"gripper_vel_xy": fingertip_lin_table_frame[:2]}

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()
