from robot_gym.core import BaseTask
from robot_gym.core.controllers import Controller
from robot_gym.core.rewards import Reward
from robot_gym.core.sensors import Sensor
from robot_gym.environment import Object, Robot, Environment
from typing import Iterable, Optional, TypeVar, Generic, Dict, Tuple
import numpy as np

from transformation import Transformation

from .logger import logger

EnvironmentType = TypeVar("EnvironmentType", bound=Environment)
ObjectType = TypeVar("ObjectType", bound=Object)


class SisyphusTask(BaseTask[EnvironmentType], Generic[EnvironmentType, ObjectType]):
    def __init__(self, controllers: Iterable[Controller],
                 sensors: Iterable[Sensor["SisyphusTask"]], reward: Reward["SisyphusTask"],
                 time_step: float = 0.005, time_limit_steps: Optional[int] = None,
                 barrier_width: float = 0.01, ball_radius: float = 0.02, done_on_border_contact: bool = False,
                 done_on_ball_lost: bool = False, max_time_steps: int = 50,
                 fingertip_pose_gripper_frame: Optional[Transformation] = None):
        """
        :param controllers:             A sequence of controller objects that define the actions on the environment
        :param sensors:                 A sequence of sensors objects that provide observations to the agent
        :param reward:                  The reward function for this task.
        :param time_step:               The time between two controller updates (actions of the agent)
        :param time_limit_steps:        The number of steps until the episode terminates (if no other termination
                                        criterion is reached)
        """
        super(SisyphusTask, self).__init__(
            controllers, sensors, reward, time_step, time_limit_steps=time_limit_steps)
        self.__barrier_width = barrier_width

        self.__target_zone_extents = np.array([0.08, 0.05])
        self.__ball_radius = ball_radius
        self._robot: Optional[Robot] = None
        self.__table_height = 0.71
        self.__fingertip_z_dist_to_ball_center = -0.01
        self.__fingertip_distance_to_table = self.__ball_radius + self.__fingertip_z_dist_to_ball_center
        self.__fingertip_pose_gripper_frame = \
            Transformation() if fingertip_pose_gripper_frame is None else fingertip_pose_gripper_frame

        self._table_extents = None
        self._table_top_center_pose = None
        self._ball_2d_position_table_frame: Optional[np.ndarray] = None
        self._ball_2d_velocity_table_frame: Optional[np.ndarray] = None
        self._current_target_vel_lin_table_frame: Optional[np.ndarray] = None
        self._current_target_vel_ang_table_frame: Optional[float] = None
        self.__current_time_step = None
        self.__done_on_border_contact = done_on_border_contact
        self.__done_on_ball_lost = done_on_ball_lost
        self.__max_time_steps = max_time_steps
        self.__rotated_fingertip_tcp_frame = self.__fingertip_pose_gripper_frame.transform(
            Transformation.from_pos_euler(euler_angles=np.array([0.0, np.pi, 0]), sequence="XYZ"))

    def reset(self) -> Dict[str, np.ndarray]:
        self._current_target_vel_lin_table_frame = np.zeros(2)
        self._current_target_vel_ang_table_frame = 0.0
        self.__current_time_step = 0
        return super(SisyphusTask, self).reset()

    def _step_task(self) -> Tuple[bool, Dict]:
        self.__current_time_step += 1
        done = False
        if self.__current_time_step >= self.__max_time_steps:
            logger.info("Episode done because time is up.")
            done = True
        if self.__done_on_border_contact and not done:
            dist = self.table_top_finger_limits - np.abs(self.finger_pose_table_frame.translation[:2])
            done = np.min(dist) < 0.005
            if done:
                logger.info("Episode done because of contact with border.")
        if self.__done_on_ball_lost and not done:
            diff = self.ball_2d_position_table_frame[1] - self.finger_pose_table_frame.translation[1]
            done = diff <= -0.03  # Ball is 3cm below the gripper
            if done:
                logger.info("Episode done because ball has been lost.")
        if done:
            logger.info("")
        return done, {}

    @property
    def target_zone_pos_table_frame(self) -> np.ndarray:
        return np.array([0, self.table_top_finger_limits[1] - self.__target_zone_extents[1] / 2])

    @property
    def target_zone_extents(self) -> np.ndarray:
        return self.__target_zone_extents

    @property
    def ball_2d_position_table_frame(self) -> Optional[np.ndarray]:
        return self._ball_2d_position_table_frame

    @property
    def ball_2d_velocity_table_frame(self) -> Optional[np.ndarray]:
        return self._ball_2d_velocity_table_frame

    @property
    def table_extents(self) -> np.ndarray:
        return self._table_extents

    @property
    def table_top_accessible_extents(self):
        return self.table_extents - self.__barrier_width * 2

    @property
    def table_top_finger_limits(self):
        return self.table_top_accessible_extents / 2 - self.ball_radius * 2 - 0.04

    @property
    def table_top_center_pose(self) -> Transformation:
        return self._table_top_center_pose

    @property
    def fingertip_distance_to_table(self) -> float:
        return self.__fingertip_distance_to_table

    @property
    def barrier_width(self) -> float:
        return self.__barrier_width

    @property
    def ball_radius(self) -> float:
        return self.__ball_radius

    @property
    def fingertip_pose_gripper_frame(self):
        return self.__fingertip_pose_gripper_frame

    @property
    def current_target_vel_lin_table_frame(self) -> Optional[np.ndarray]:
        return self._current_target_vel_lin_table_frame

    @property
    def current_target_vel_ang_table_frame(self) -> Optional[float]:
        return self._current_target_vel_ang_table_frame

    def render(self, mode="human"):
        assert mode == "human"

    @property
    def finger_pose_table_frame(self) -> Transformation:
        return self._table_top_center_pose.inv * self._robot.gripper.pose * self.__rotated_fingertip_tcp_frame

    @property
    def rotated_fingertip_pose_tcp_frame(self) -> Transformation:
        return self.__rotated_fingertip_tcp_frame
