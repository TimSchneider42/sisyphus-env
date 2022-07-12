from typing import Dict, List, Union, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from robot_gym.core.controllers import Controller
from transformation import Transformation
from .sisyphus_task import SisyphusTask
from .logger import logger


def inner_ray_intersect(
        lower_lims: np.ndarray, upper_lims: np.ndarray, ray_origin: np.ndarray, ray_dir: np.ndarray) -> np.ndarray:
    lims = np.stack([lower_lims, upper_lims])
    dist_rel = np.divide(lims - ray_origin[np.newaxis], ray_dir[np.newaxis], out=np.full((2, 2), -1.0),
                         where=ray_dir[np.newaxis] != 0)
    if np.all(dist_rel < 0):
        return np.full(2, np.nan)
    dist_rel_min = np.min(dist_rel[dist_rel >= 0])
    return ray_origin + dist_rel_min * ray_dir


class SisyphusEndEffectorVelocityController(Controller[SisyphusTask]):
    def __init__(self, robot_name: str, linear_limits_lower: np.ndarray, linear_limits_upper: np.ndarray,
                 angular_limit_lower: float, angular_limit_upper: float, rotation_range: float = 0.3,
                 control_lin_accel: bool = False, max_lin_accel: float = 1.0, disable_rotation: bool = False):
        super().__init__("sisyphus_end_effector_velocity_controller")
        self.__linear_limits_lower = linear_limits_lower
        self.__linear_limits_upper = linear_limits_upper
        self.__angular_limit_lower = angular_limit_lower
        self.__angular_limit_upper = angular_limit_upper

        self.__robot_name = robot_name
        self.__rotation_range = rotation_range
        self.__control_lin_accel = control_lin_accel
        self.__max_lin_accel = max_lin_accel
        self.__disable_rotation = disable_rotation
        self.__outer_border_margin = 0.005

    def _actuate_denormalized(self, action: np.ndarray):
        task = self.task
        dt = self.task.time_step
        gripper = self.__robot.gripper
        fingertip_rotation = Transformation.from_pos_euler(euler_angles=np.array([0.0, np.pi, 0]), sequence="XYZ")
        rotated_fingertip_pose = task.fingertip_pose_gripper_frame * fingertip_rotation
        current_fingertip_pose = gripper.pose * rotated_fingertip_pose
        fingertip_pose_table_frame = task.table_top_center_pose.transform(current_fingertip_pose, inverse=True)
        fingertip_vel_table_frame = task.table_top_center_pose.rotation.apply(gripper.velocity)
        fingertip_lin_vel_table_frame, fingertip_ang_vel_table_frame = fingertip_vel_table_frame
        fingertip_pos = fingertip_pose_table_frame.translation[:2]
        lim = task.table_top_finger_limits

        if self.__control_lin_accel:
            current_vel = fingertip_lin_vel_table_frame[:2]
            target_linear_vel_xy = current_vel + action[:2] * dt
            linear_vel_xy = np.clip(target_linear_vel_xy, self.__linear_limits_lower, self.__linear_limits_upper)
        else:
            linear_vel_xy = action[:2]

        if np.linalg.norm(linear_vel_xy) > 0:
            # Add a bit of margin to the limits to allow movement along the sides
            target_pos_2d = fingertip_pos + inner_ray_intersect(
                -lim - self.__outer_border_margin, lim + self.__outer_border_margin, fingertip_pos, linear_vel_xy)
        else:
            target_pos_2d = fingertip_pos
        target_pos_2d = np.clip(target_pos_2d, -lim, lim)
        target_position_table_frame = np.concatenate([target_pos_2d, [task.fingertip_distance_to_table]])

        if not self.__disable_rotation:
            angular_vel_z = action[2]
        else:
            angular_vel_z = 0.1  # Allow for error correction

        self.task._current_target_vel_lin_table_frame = linear_vel_xy
        self.task._current_target_vel_ang_table_frame = angular_vel_z

        current_angle = fingertip_pose_table_frame.rotation.as_euler("XYZ")[2]

        if angular_vel_z == 0:
            target_rotation_z = current_angle
        else:
            target_rotation_z = np.sign(angular_vel_z) * self.__rotation_range
        target_rotation = Rotation.from_euler("xyz", [0, 0, target_rotation_z])

        # TODO: can small errors in XY accumulate over time if the angular velocity is set to 0 for too long?
        if current_angle > self.__rotation_range:
            angular_vel_z = max(np.abs(angular_vel_z), 0.2 * self.__angular_limit_upper)

        target_fingertip_pose_table_frame = Transformation(target_position_table_frame, target_rotation)
        target_tcp_pose_table_frame = target_fingertip_pose_table_frame * rotated_fingertip_pose.inv
        target_tcp_pose_world_frame = task.table_top_center_pose.transform(target_tcp_pose_table_frame)

        self.__robot.arm.move_towards_pose_linear(target_tcp_pose_world_frame, np.linalg.norm(linear_vel_xy),
                                                  np.abs(angular_vel_z))

    def _initialize(self, task: SisyphusTask) -> Tuple[np.ndarray, np.ndarray]:
        if np.any(task.fingertip_pose_gripper_frame.translation[:2] != 0):
            logger.warning(
                "Warning: using SisyphusEndEffectorVelocityController with a XY TCP->fingertip translation other 0."
                " This will lead to velocity and positioning errors.")
            # The reason for this is that a nonzero XY translation between TCP and fingertip means that angular velocity
            # of the TCP will translate into linear velocity of the fingertip. Hence, the fingertip will not hold its
            # set linear velocity.
        self.__robot = self.task.environment.robots[self.__robot_name]
        if self.__control_lin_accel:
            lin_lim_lower = np.full(2, -self.__max_lin_accel)
            lin_lim_upper = np.full(2, self.__max_lin_accel)
        else:
            lin_lim_lower = self.__linear_limits_lower
            lin_lim_upper = self.__linear_limits_upper
        if not self.__disable_rotation:
            return np.concatenate([lin_lim_lower, [self.__angular_limit_lower]]), \
                   np.concatenate([lin_lim_upper, [self.__angular_limit_upper]])
        else:
            return np.concatenate([lin_lim_lower]), np.concatenate([lin_lim_upper])
