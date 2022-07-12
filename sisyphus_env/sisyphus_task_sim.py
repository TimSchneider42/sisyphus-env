from abc import abstractmethod

from pathlib import Path
from tempfile import NamedTemporaryFile

from robot_gym.core.controllers import Controller
from robot_gym.core.rewards import Reward
from robot_gym.core.sensors import Sensor
from robot_gym.environment.simulation import SimulationObject, ShapeTypes, SimulationRobot, SimulationEnvironment
from typing import Tuple, Dict, Iterable, Optional, Sequence
import numpy as np

from .sisyphus_task import SisyphusTask
from transformation import Transformation


class SisyphusTaskSim(SisyphusTask[SimulationEnvironment, SimulationObject]):
    def __init__(self, controllers: Iterable[Controller],
                 sensors: Iterable[Sensor["SisyphusTaskSim"]], rewards: Reward["SisyphusTaskSim"],
                 time_step: float = 0.005, time_limit_steps: Optional[int] = None, use_ball: bool = True,
                 gripper_width: float = 0.02, ball_friction: float = 1.0, ball_mass: float = 0.02,
                 ball_restitution: float = 0.0, robot_pose: Optional[Transformation] = None):
        """
        :param controllers:             A sequence of controller objects that define the actions on the environment
        :param sensors:                 A sequence of sensors objects that provide observations to the agent
        :param rewards:                 A sequence of rewards objects that provide rewards to the agent
        :param time_step:               The time between two controller updates (actions of the agent)
        :param time_limit_steps:        The number of steps until the episode terminates (if no other termination
                                        criterion is reached)
        """
        fingertip_pose_gripper_frame = Transformation([0, 0, 0.08])
        super(SisyphusTaskSim, self).__init__(
            controllers, sensors, rewards, time_step, time_limit_steps=time_limit_steps, barrier_width=0.02,
            ball_radius=0.02, fingertip_pose_gripper_frame=fingertip_pose_gripper_frame)
        self._table_extents = None
        self._table_top_center_pose = None
        self.__target_zone_height = 0.001
        self._robot: Optional[SimulationRobot] = None
        self.__gripper_z_dist_to_ball_center = 0.005
        self.__gripper_distance_to_table = self.ball_radius + self.__gripper_z_dist_to_ball_center
        self.__use_ball = use_ball
        self.__gripper_width = gripper_width
        self.__ball_friction = ball_friction
        self.__ball_mass = ball_mass
        self.__ball_restitution = ball_restitution
        self.__ball = None
        if robot_pose is None:
            robot_pose = Transformation()
        self.__robot_pose = robot_pose
        self.initial_ball_pos_override: Optional[np.ndarray] = None

    def _add_box(self, extents: Sequence[float], pose: Transformation,
                 rgba_colors: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0), friction: float = 0.5):
        collision_shape = self.environment.create_collision_shape(
            ShapeTypes.BOX, box_extents=[extents])
        visual_shape = self.environment.create_visual_shape(
            ShapeTypes.BOX, box_extents=[extents], rgba_colors=rgba_colors)
        box = self.environment.add_simple_object(visual_shape, collision_shape, friction=friction)
        box.set_pose(pose)

    def _initialize(self):
        with (Path(__file__).parent / "cuboid_gripper_template.urdf").open() as f:
            gripper_urdf_template = f.read()
        gripper_urdf_data = gripper_urdf_template.format(gripper_width=self.__gripper_width)
        with NamedTemporaryFile("w", suffix=".urdf") as f:
            f.write(gripper_urdf_data)
            f.flush()
            self._robot = self.environment.add_ur10_robot("ur10", rh_p12_rn_urdf_path=Path(f.name))
        self._robot.set_pose(self.__robot_pose * Transformation.from_pos_euler(position=np.array([0.0, 0.0, 0.71])))
        # Robot socket
        self._add_box([0.15, 0.15, 0.71], self.__robot_pose * Transformation(np.array([0, 0, 0.355])),
                      rgba_colors=(0.7, 0.7, 0.7, 1.0))

        self._table_extents, self._table_top_center_pose = self._setup_table()

        if self.__use_ball:
            ball_collision_shape = self.environment.create_collision_shape(
                ShapeTypes.SPHERE, sphere_radii=[self.ball_radius])
            ball_visual_shape = self.environment.create_visual_shape(
                ShapeTypes.SPHERE, sphere_radii=[self.ball_radius], rgba_colors=(0.7, 0.2, 0.2, 1.0))
            self.__ball = self.environment.add_simple_object(
                ball_visual_shape, ball_collision_shape, mass=self.__ball_mass, friction=self.__ball_friction,
                restitution=self.__ball_restitution)
            self.__ball.set_pose(
                self._table_top_center_pose * Transformation.from_pos_euler(position=[0, 0, self.ball_radius + 0.001]))

        target_zone_marker_visual = self.environment.create_visual_shape(
            ShapeTypes.BOX, box_extents=[list(self.target_zone_extents) + [self.__target_zone_height]],
            rgba_colors=(1.0, 0.0, 0.0, 0.2))
        target_zone_marker = self.environment.add_simple_object(target_zone_marker_visual)
        target_zone_pose_table_top_frame = Transformation.from_pos_euler(
            np.concatenate([self.target_zone_pos_table_frame, [self.__target_zone_height / 2]]))
        target_zone_marker.set_pose(self._table_top_center_pose * target_zone_pose_table_top_frame)

        gripper_pose_table_top_frame = Transformation.from_pos_euler(
            position=np.array([0, 0, self.__gripper_distance_to_table]), euler_angles=[np.pi, 0, 0])
        gripper_pose_tcp_world_frame = self._table_top_center_pose.transform(gripper_pose_table_top_frame)
        self._robot.arm.move_to_pose(gripper_pose_tcp_world_frame)
        self.environment.set_reset_checkpoint()

    def __refresh_ball_position(self):
        if self.__use_ball:
            self._ball_2d_position_table_frame = self.table_top_center_pose.transform(
                self.__ball.pose, inverse=True).translation[:2]
            self._ball_2d_velocity_table_frame = self.table_top_center_pose.rotation.apply(
                self.__ball.velocity[0], inverse=True)[:2]

    def _step_task(self) -> Tuple[bool, Dict]:
        self.__refresh_ball_position()
        return super(SisyphusTaskSim, self)._step_task()

    def _reset_fingertip_pos(self, pos_table_frame: np.ndarray):
        fingertip_pose_table_frame = Transformation.from_pos_euler(
            np.concatenate([pos_table_frame, [self.ball_radius + self.__gripper_z_dist_to_ball_center]]),
            np.array([0.0, np.pi, 0.0]))
        gripper_pose_table_frame = fingertip_pose_table_frame * self.fingertip_pose_gripper_frame.inv
        fingertip_pose_world_frame = self.table_top_center_pose.transform(gripper_pose_table_frame)
        self._robot.arm.move_to_pose(fingertip_pose_world_frame)

    def _reset_task(self):
        if self.initial_ball_pos_override is None:
            initial_ball_pos = self._get_initial_ball_pos()
        else:
            initial_ball_pos = self.initial_ball_pos_override
            self.initial_ball_pos_override = None
        initial_ball_pos_table_frame = np.array([0.0, 0.0, self.ball_radius + 0.001])
        initial_ball_pos_table_frame[:2] += initial_ball_pos
        initial_ball_pose_table_frame = Transformation.from_pos_euler(position=initial_ball_pos_table_frame)
        initial_ball_pose_world_frame = self._table_top_center_pose * initial_ball_pose_table_frame
        if self.__use_ball:
            self.__ball.set_pose(initial_ball_pose_world_frame)
        self._reset_fingertip_pos(initial_ball_pos_table_frame[:2] + np.array([0, -0.03]))
        self.__refresh_ball_position()

    @abstractmethod
    def _setup_table(self):
        pass

    @abstractmethod
    def _get_initial_ball_pos(self):
        pass
