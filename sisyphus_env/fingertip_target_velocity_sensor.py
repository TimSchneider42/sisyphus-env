from typing import Dict, Tuple, List

import numpy as np

from robot_gym.core.sensors import ContinuousSensor
from .sisyphus_task import SisyphusTask


class FingertipTargetVelocity2DSensor(ContinuousSensor[SisyphusTask]):
    def __init__(self, linear_limit_lower: np.ndarray, linear_limit_upper: np.ndarray,
                 angular_limit_lower: float, angular_limit_upper: float, sense_angle: bool = True):
        super(FingertipTargetVelocity2DSensor, self).__init__()
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
                "gripper_target_vel_xy": lin_lims,
                "gripper_target_angular_vel_z": ang_lims
            }
        else:
            return {
                "gripper_target_vel_xy": lin_lims
            }

    def __observe(self) -> Dict[str, np.ndarray]:
        if self.__sense_angle:
            return {
                "gripper_target_vel_xy": self.task.current_target_vel_lin_table_frame,
                "gripper_target_angular_vel_z": np.array([self.task.current_target_vel_ang_table_frame])
            }
        else:
            return {"gripper_target_vel_xy": self.task.current_target_vel_lin_table_frame}

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__observe()
