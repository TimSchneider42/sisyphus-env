import time

from robot_gym.core.controllers import Controller
from robot_gym.core.rewards import Reward
from robot_gym.core.sensors import Sensor
from typing import Tuple, Iterable, Optional, Dict, Any
import numpy as np

from transformation import Transformation

from robot_gym.environment.pybullet import PybulletEnvironment
from .logger import logger
from .sisyphus_task_sim import SisyphusTaskSim


class SisyphusTaskSimRealistic(SisyphusTaskSim):
    def __init__(self, controllers: Iterable[Controller],
                 sensors: Iterable[Sensor["SisyphusTaskSim"]], rewards: Reward["SisyphusTaskSim"],
                 time_step: float = 0.005, time_limit_steps: Optional[int] = None, use_ball: bool = True,
                 gripper_width: float = 0.028, ball_friction: float = 0.5, ball_mass: float = 0.03,
                 table_extents: Tuple[float, float] = (0.5, 0.57), pickup_hole_width: float = 0.03,
                 pickup_hole_pos_y: float = 0.2, ball_restitution: float = 0.4,
                 table_pose_robot_frame: Optional[Transformation] = None, obs_to_act_time: float = 0.2,
                 act_time: float = 0.02):
        """
        :param controllers:             A sequence of controller objects that define the actions on the environment
        :param sensors:                 A sequence of sensors objects that provide observations to the agent
        :param rewards:                 A sequence of rewards objects that provide rewards to the agent
        :param time_step:               The time between two controller updates (actions of the agent)
        :param time_limit_steps:        The number of steps until the episode terminates (if no other termination
                                        criterion is reached)
        """
        super(SisyphusTaskSimRealistic, self).__init__(
            controllers, sensors, rewards, time_step, use_ball=use_ball, gripper_width=gripper_width,
            ball_friction=ball_friction, ball_mass=ball_mass, time_limit_steps=time_limit_steps,
            ball_restitution=ball_restitution,
            robot_pose=Transformation.from_pos_euler(euler_angles=[0.0, 0.0, -np.pi / 2]))
        self.__table_extents = np.array(table_extents)
        self.__target_zone_height = 0.001
        self.__barrier_height = 0.02
        self.__table_base_color = (0.4, 0.3, 0.3, 1.0)
        self.__surface_box_color = (0.5, 0.4, 0.4, 1.0)
        self.__boundary_color = (0.3, 0.2, 0.2, 1.0)
        self.__table_height = 0.02
        self.__pickup_hole_width = pickup_hole_width
        self.__pickup_hole_dist_y = pickup_hole_pos_y
        self.__rail_length = 0.15
        self.__rail_width = 0.015
        self.__previous_step: Optional[float] = None
        self.__obs_to_act_time = obs_to_act_time
        self.__base_simulator_step: Optional[float] = None
        self.__act_time = act_time
        if table_pose_robot_frame is None:
            table_pose_robot_frame = Transformation.from_pos_euler(
                [-0.8811958671743889, -0.06563322473695679, -0.06650364306714512],
                [0.27121187, -0.00205137, 1.60157772])
        self.__table_top_center_pose_robot_frame = table_pose_robot_frame

    def _setup_table(self):
        # Table
        bw = self.barrier_width
        bh = self.__barrier_height

        barrier_friction = 0.0
        base_friction = 0.1

        table_top_center_pose = self.environment.robots["ur10"].pose * self.__table_top_center_pose_robot_frame

        y_extension = self.__rail_length + 0.02
        collector_half_box_extents = np.array(
            [(self.__table_extents[0] - self.__rail_width) / 2, y_extension, self.__table_height])
        main_base_box_extents = np.concatenate(
            [self.__table_extents + np.array([0.0, self.__pickup_hole_dist_y - self.__rail_length]),
             [self.__table_height]])
        table_top_table_frame = Transformation.from_pos_euler(
            position=[0, (self.__pickup_hole_dist_y - self.__rail_length) / 2, self.__table_height / 2])
        table_pose = table_top_center_pose * table_top_table_frame.inv

        # Main base
        self._add_box(main_base_box_extents, table_pose, rgba_colors=self.__surface_box_color, friction=base_friction)

        # Left base
        self._add_box(
            collector_half_box_extents, table_pose * Transformation(
                [-(collector_half_box_extents[0] + self.__rail_width) / 2,
                 -(main_base_box_extents[1] + collector_half_box_extents[1]) / 2, 0.0]),
            rgba_colors=self.__surface_box_color, friction=base_friction)

        # Right base
        self._add_box(
            collector_half_box_extents, table_pose * Transformation(
                [(collector_half_box_extents[0] + self.__rail_width) / 2,
                 -(main_base_box_extents[1] + collector_half_box_extents[1]) / 2, 0.0]),
            rgba_colors=self.__surface_box_color, friction=base_friction)

        rail_end_box_length = np.sqrt(2) * self.__rail_width / 2
        rail_end_box_extents = np.array(
            [rail_end_box_length,
             np.sqrt((self.__rail_width / 2) ** 2 - (rail_end_box_length / 2) ** 2),
             self.__table_height])
        self._add_box(
            rail_end_box_extents, table_pose * Transformation.from_pos_euler(
                [-self.__rail_width / 2 + 1 / np.sqrt(2) * rail_end_box_extents[1] / 2,
                 -main_base_box_extents[1] / 2 - 1 / np.sqrt(2) * rail_end_box_extents[1] / 2, 0.0],
                [0.0, 0.0, np.pi / 4]),
            rgba_colors=self.__surface_box_color, friction=base_friction)
        self._add_box(
            rail_end_box_extents, table_pose * Transformation.from_pos_euler(
                [self.__rail_width / 2 - 1 / np.sqrt(2) * rail_end_box_extents[1] / 2,
                 -main_base_box_extents[1] / 2 - 1 / np.sqrt(2) * rail_end_box_extents[1] / 2, 0.0],
                [0.0, 0.0, -np.pi / 4]),
            rgba_colors=self.__surface_box_color, friction=base_friction)

        # Boundaries
        self._add_box(
            [self.__table_extents[0], bw, bh],
            table_top_center_pose * Transformation.from_pos_euler(
                position=[0, self.__table_extents[1] / 2 - bw / 2, bh / 2]),
            rgba_colors=self.__boundary_color, friction=barrier_friction)
        self._add_box(
            [bw, self.__table_extents[1] - bw, bh],
            table_top_center_pose * Transformation.from_pos_euler(
                position=[-(self.__table_extents[0] / 2 - bw / 2), -bw / 2, bh / 2]),
            rgba_colors=self.__boundary_color, friction=barrier_friction)
        self._add_box(
            [bw, self.__table_extents[1] - bw, bh],
            table_top_center_pose * Transformation.from_pos_euler(
                position=[self.__table_extents[0] / 2 - bw / 2, -bw / 2, bh / 2]),
            rgba_colors=self.__boundary_color, friction=barrier_friction)

        # Collector boundaries
        pickup_hole_dist_x = (self.__table_extents[0] - 2 * bw - self.__pickup_hole_width) / 2
        cb_extents = np.array(
            [np.sqrt(pickup_hole_dist_x ** 2 + self.__pickup_hole_dist_y ** 2), bw, bh * 2])
        cb_left_top = np.array([-self.__pickup_hole_width / 2 - pickup_hole_dist_x, -self.__table_extents[1] / 2])
        cb_left_right = np.array(
            [-self.__pickup_hole_width / 2, -self.__table_extents[1] / 2 - self.__pickup_hole_dist_y])
        cb_left_dir = (cb_left_top - cb_left_right) / np.linalg.norm(cb_left_top - cb_left_right)
        cb_left_ort = np.array([-cb_left_dir[1], cb_left_dir[0]])
        cb_left_pos2d = (cb_left_top + cb_left_right) / 2 + cb_left_ort * bw / 2
        cb_left_angle = np.arctan2(cb_left_dir[1], cb_left_dir[0])
        cb_left_pose = Transformation.from_pos_euler(
            np.concatenate([cb_left_pos2d, [bh]]), [0.0, 0.0, cb_left_angle])

        cb_right_pos2d = cb_left_pos2d.copy()
        cb_right_pos2d[0] *= -1
        cb_right_pose = Transformation.from_pos_euler(
            np.concatenate([cb_right_pos2d, [bh]]), [0.0, 0.0, -cb_left_angle])

        self._add_box(cb_extents, table_top_center_pose * cb_left_pose, rgba_colors=self.__boundary_color,
                      friction=barrier_friction)
        self._add_box(cb_extents, table_top_center_pose * cb_right_pose, rgba_colors=self.__boundary_color,
                      friction=barrier_friction)
        return self.__table_extents, table_top_center_pose

    def _get_initial_ball_pos(self):
        initial_ball_pos_y = -self._table_extents[1] / 2 + 0.1 + 0.03
        return np.random.normal([0.0, initial_ball_pos_y], [0.01, 0.0])

    def _reset_task(self):
        super(SisyphusTaskSimRealistic, self)._reset_task()
        self.__previous_step = time.time()

    def _initialize(self):
        self.__base_simulator_step = self.environment.time_step / self.environment.substeps_per_step
        super(SisyphusTaskSimRealistic, self)._initialize()

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        time_spent_computing_action = time.time() - self.__previous_step
        if time_spent_computing_action > self.__obs_to_act_time:
            logger.warning("Missed action timing by {:0.2f}ms.".format(
                (time_spent_computing_action - self.__obs_to_act_time) * 1000))
        time_passed = max(time_spent_computing_action, self.__obs_to_act_time) + self.__act_time
        environment = self.environment
        assert isinstance(environment, PybulletEnvironment)
        simulator_steps = round(time_passed / self.__base_simulator_step)
        rounded_time = simulator_steps * self.__base_simulator_step
        environment.reconfigure_time_step(new_time_step=rounded_time, new_substeps_per_step=simulator_steps,
                                          new_virtual_step_mode=environment.virtual_substep_mode)
        environment.step()
        act_to_obs_time = self.time_step - self.__obs_to_act_time - self.__act_time
        simulator_steps = round(act_to_obs_time / self.__base_simulator_step)
        rounded_time = simulator_steps * self.__base_simulator_step
        environment.reconfigure_time_step(new_time_step=rounded_time, new_substeps_per_step=simulator_steps,
                                          new_virtual_step_mode=environment.virtual_substep_mode)
        output = super(SisyphusTaskSimRealistic, self).step(action)
        self.__previous_step = time.time()
        return output
