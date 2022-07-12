from datetime import datetime
import time
from pathlib import Path
from queue import Queue
import traceback

import pybullet_data

from robot_gym.core.base_task import EpisodeInvalidException, WrappedEpisodeInvalidException
from robot_gym.core.controllers import Controller
from robot_gym.core.rewards import Reward
from robot_gym.core.sensors import Sensor
from .logger import logger
from natnet_client import DataFrame, LabeledMarker
from rhp12rn import DynamixelCommunicationError, DynamixelPacketError
from scipy.spatial.transform import Rotation

from robot_gym.environment import Robot, Object
from pyboolet import PhysicsClient
from pyboolet.multibody import Box, URDFBody, Multibody, Sphere
from robot_gym.environment.real import UR10RobotArm, RHP12RNGripper, RealEnvironment, OptitrackRigidBody, \
    UR10ConnectionLostException
from typing import Tuple, Dict, Iterable, Optional, List, Any
import numpy as np

from .sisyphus_task import SisyphusTask
from .point_tracker import PointTracker
from transformation import Transformation


class CriticalSafetyException(Exception):
    pass


class SafetyException(EpisodeInvalidException):
    pass


class BallTrackingException(EpisodeInvalidException):
    pass


class SisyphusTaskReal(SisyphusTask[RealEnvironment, Object]):
    def __init__(self, controllers: Iterable[Controller],
                 sensors: Iterable[Sensor["SisyphusTaskReal"]], rewards: Reward["SisyphusTaskReal"],
                 time_step: float = 0.005, time_limit_steps: Optional[int] = None, headless: bool = True,
                 finger_rotation_range: float = 0.3, max_time_steps: int = 50, use_ball: bool = False,
                 done_on_border_contact: bool = False, done_on_ball_lost: bool = False,
                 log_dir: Optional[Path] = None):
        """
        :param controllers:             A sequence of controller objects that define the actions on the environment
        :param sensors:                 A sequence of sensors objects that provide observations to the agent
        :param rewards:                 A sequence of rewards objects that provide rewards to the agent
        :param time_step:               The time between two controller updates (actions of the agent)
        :param time_limit_steps:        The number of steps until the episode terminates (if no other termination
                                        criterion is reached)
        """
        # This is relative to the old TCP (0.0, 0.0, 183.0), (0.0, 0.0, 0.31)
        # fingertip_pose_tcp_frame = Transformation.from_pos_euler(
        #     np.array([-0.002, -0.0485, 0.062]), np.array([0.05, 0.0, 0.0]))
        fingertip_pose_tcp_frame = Transformation()
        super(SisyphusTaskReal, self).__init__(
            controllers, sensors, rewards, time_step, time_limit_steps=time_limit_steps, barrier_width=0.02,
            ball_radius=0.02, fingertip_pose_gripper_frame=fingertip_pose_tcp_frame,
            done_on_ball_lost=done_on_ball_lost, done_on_border_contact=done_on_border_contact,
            max_time_steps=max_time_steps)
        self.__target_zone_extents = np.array([0.08, 0.05])
        self.__target_entry_depth = 0.02
        self._table_extents = None
        self._table_top_center_pose: Optional[Transformation] = None
        self._table_top_center_initial_pose: Optional[Transformation] = None
        self._robot: Optional[Robot[UR10RobotArm, RHP12RNGripper]]
        self.__table_rigid_body: Optional[OptitrackRigidBody] = None
        self.__table_upper_boundary_rigid_body: Optional[OptitrackRigidBody] = None
        self.__table_center_pose_rb_frame = None
        self.__pickup_location_offset_y = -0.17
        self.__boundary_begins_offset_y = -0.12
        self.__initial_gripper_pos_offset_y = 0.1
        self.__pickup_correction_x = 0.001
        self.__headless = headless
        self.__ball_tracker: Optional[PointTracker] = None
        self.__optitrack_frame_queue = Queue()

        self.__pb_table: Optional[Multibody] = None
        self.__pb_robot: Optional[URDFBody] = None
        self.__used_pb_optitrack_markers: Optional[Dict[int, Multibody]] = None
        self.__unused_pb_optitrack_markers: Optional[List[Multibody]] = None
        self.__pb_ball: Optional[Multibody] = None
        self.__pb_ground_plane: Optional[Multibody] = None
        if self.__headless:
            self.__pybullet = None
        else:
            self.__pybullet = PhysicsClient()
        self.__last_ball_sighting = None
        self.__finger_transfer_height = 0.1
        self.__optitrack_gripper: Optional[OptitrackRigidBody] = None
        self.__enable_sanity_check = False
        self.__finger_rotation_range = finger_rotation_range
        self.__check_ball_state = False
        self.__robot_has_moved = False
        self.__use_ball = use_ball
        self.__gripper_pos_off_since = 0
        self.__gripper_ang_off_since = 0
        self.__during_episode = False
        self.__resetting_start_time = None
        self.__log_dir = log_dir

    def _initialize(self):
        self.__during_episode = False
        self.environment.synchronized_mode = False
        self.__optitrack_gripper = self.environment.optitrack.rigid_bodies["gripper"]

        # Determine table properties
        self.__table_rigid_body = self.environment.optitrack.rigid_bodies["table"]
        self.__table_upper_boundary_rigid_body = self.environment.optitrack.rigid_bodies["table_upper_boundary"]
        self.__table_center_pose_rb_frame, self._table_extents = self._determine_table_properties(
            self.__table_rigid_body, self.__table_upper_boundary_rigid_body, self.barrier_width)
        logger.info("Determined table extents to be {:0.2f}mm x {:0.2f}mm.".format(*(self._table_extents * 1000)))
        self.__refresh_table_pose()
        logger.info("Table pose is {}.".format(self._table_top_center_pose))

        if self.__use_ball:
            self.__ball_tracker = PointTracker(
                dimensions=2, observation_noise=0.001, position_noise_per_s=0.001, velocity_noise_per_s=1.0)
            ball_candidates = self.__find_ball_candidates(
                self.environment.optitrack.markers, coordinates_in_optitrack_frame=False)
            assert len(ball_candidates) == 1, "Expected number of balls on the table to be 1, got {}.".format(
                len(ball_candidates))
            self.__last_ball_sighting = self.environment.optitrack.marker_update_timestamp
            self.__ball_tracker.add_observation(ball_candidates[0], self.environment.optitrack.marker_update_timestamp)
            self.__refresh_ball_state()
        self.environment.optitrack.on_data_frame_received_event.handlers.append(self.__on_optitrack_frame_received)
        self.__target_zone_pos_table_frame = np.array(
            [0, (self._table_extents[1] - self.__target_zone_extents[1] - 0.05) / 2 - self.barrier_width])
        self._robot = self.environment.robots["ur10"]
        gripper = self._robot.gripper
        gripper.observations_enabled = True
        gripper.joint_target_positions = gripper.joint_intervals[:, 1]
        while np.max(np.abs(gripper.joint_positions - gripper.joint_intervals[:, 1])) > 0.05:
            self.environment.step()
            time.sleep(0.1)
        self.environment.step()
        gripper.observations_enabled = False  # Disable gripper observations to save time during observation phase

        if not self.__headless:
            logger.info("Loading visualization...")
            self.__pybullet.connect_gui()
            with self.__pybullet.as_default():
                table_height = 0.005
                self.__pb_table = Box(self._table_extents.tolist() + [table_height], rgba_color=(0.4, 0.3, 0.3, 1.0),
                                      create_collision_shape=False)
                self.__pb_table.reset_pose(
                    self._table_top_center_pose.transform(Transformation([0, 0, table_height / 2])))
                self.__pb_robot = URDFBody(Path(__file__).parents[3] / "pybullet" / "models" / "ur10_real.urdf",
                                           use_fixed_base=True, base_pose=self._robot.pose)
                self.__pb_optitrack_markers = {}
                if self.__use_ball:
                    self.__pb_ball = Sphere(
                        self.ball_radius, rgba_color=(0.7, 0.2, 0.2, 1.0), create_collision_shape=False)
                self.__unused_pb_optitrack_markers = [
                    Sphere(0.01, rgba_color=(1.0, 1.0, 0.0, 1.0), create_collision_shape=False,
                           pose=Transformation([0, 0, -1.0])) for _ in range(50)]
                self.__used_pb_optitrack_markers = {}
                self.__pybullet.set_additional_search_path(pybullet_data.getDataPath())
                self.__pb_ground_plane = URDFBody("plane.urdf", use_fixed_base=True)
                self.refresh_visualization()
            logger.info("Visualization loaded.")

        target_pos = self.finger_pose_table_frame.translation.copy()
        target_pos[2] = self.__finger_transfer_height
        self.__move_fingertip_to(Transformation(target_pos))
        self.__move_fingertip_to(Transformation([0.0, 0.0, self.__finger_transfer_height]))
        self.environment.synchronized_mode = True
        self.__enable_sanity_check = True
        self.__robot_has_moved = False

    def refresh_visualization(self):
        if not self.__headless:
            for pb_j, j_pos in zip(self.__pb_robot.revolute_joints, self._robot.arm.joint_positions):
                pb_j.reset_joint_state(j_pos, velocity=0.0)
            env = self.environment
            active_ids = {m.id_num for m in env.optitrack.markers}
            tracked_ids = set(self.__used_pb_optitrack_markers.keys())
            for m_id in tracked_ids.difference(active_ids):
                self.__used_pb_optitrack_markers[m_id].reset_pose(Transformation([0, 0, -1.0]))
                self.__unused_pb_optitrack_markers.append(self.__used_pb_optitrack_markers[m_id])
                del self.__used_pb_optitrack_markers[m_id]
            for m in env.optitrack.markers:
                if m.id_num not in self.__used_pb_optitrack_markers:
                    if len(self.__unused_pb_optitrack_markers) > 0:
                        self.__used_pb_optitrack_markers[m.id_num] = self.__unused_pb_optitrack_markers[-1]
                        self.__unused_pb_optitrack_markers[-1:] = []
                self.__used_pb_optitrack_markers[m.id_num].reset_pose(Transformation(m.pos))
            if self.__use_ball:
                ball_pose_table_frame = Transformation(
                    np.concatenate([self.ball_2d_position_table_frame, [self.ball_radius]]))
                self.__pb_ball.reset_pose(self.table_top_center_pose * ball_pose_table_frame)

    @staticmethod
    def _determine_table_properties(
            table_rigid_body: OptitrackRigidBody, table_upper_boundary_rigid_body: OptitrackRigidBody,
            barrier_width: float) -> Tuple[Transformation, np.ndarray]:
        assert table_rigid_body.tracking_valid, "Table not found in scene."
        assert table_upper_boundary_rigid_body.tracking_valid, "Table upper boundary not found in scene."
        marker_positions = np.array([m.pos for m in table_rigid_body.description.markers])
        assert marker_positions.shape[0] == 5
        marker_distances = np.linalg.norm(marker_positions[None] - marker_positions[:, None], axis=-1)
        inf_val = np.max(marker_distances) + 1
        dist_mat_inf = marker_distances + np.eye(5) * inf_val
        closest_markers = np.unravel_index(np.argmin(dist_mat_inf), marker_distances.shape)
        dist_mat_inf[closest_markers] = inf_val
        dist_mat_inf[tuple(reversed(closest_markers))] = inf_val
        second_closest_markers = np.unravel_index(np.argmin(dist_mat_inf), marker_distances.shape)
        outside_marker = np.intersect1d(closest_markers, second_closest_markers)[0]
        left_bottom_marker = [m for m in closest_markers if m != outside_marker][0]
        right_bottom_marker = [m for m in second_closest_markers if m != outside_marker][0]
        dist_mat_inf[:, outside_marker] = inf_val
        dist_mat_inf[outside_marker, :] = inf_val
        dist_mat_inf_sorted = np.argsort(dist_mat_inf, axis=1)
        left_top_marker = dist_mat_inf_sorted[left_bottom_marker][1]
        right_top_marker = dist_mat_inf_sorted[right_bottom_marker][1]
        assert len({outside_marker, left_bottom_marker, right_bottom_marker, right_top_marker, left_top_marker}) == 5
        corner_markers = [left_bottom_marker, right_bottom_marker, right_top_marker, left_top_marker]

        # Fit plane to corner marker positions
        corner_pos = marker_positions[corner_markers]
        mean_corner_pos = np.mean(marker_positions[corner_markers], axis=0)
        cov = np.cov((corner_pos - mean_corner_pos[None]).T)
        eigen_val, eigen_vec = np.linalg.eig(cov)
        rotation_matrix = eigen_vec[:, np.argsort(eigen_val)][:, [1, 2, 0]]

        if rotation_matrix[:, 0].dot(marker_positions[right_bottom_marker] - marker_positions[left_bottom_marker]) < 0:
            rotation_matrix[:, 0] *= -1
        if rotation_matrix[:, 1].dot(marker_positions[left_top_marker] - marker_positions[left_bottom_marker]) < 0:
            rotation_matrix[:, 1] *= -1
        if rotation_matrix[:, 2].dot(np.cross(rotation_matrix[:, 0], rotation_matrix[:, 1])) < 0:
            rotation_matrix[:, 2] *= -1

        table_rotation_optitrack_frame = Rotation.from_matrix(rotation_matrix)

        center_offset = np.array([0, 0, -0.041])
        table_translation_optitrack_frame = mean_corner_pos + table_rotation_optitrack_frame.apply(center_offset)
        table_center_pose_rb_frame = Transformation(table_translation_optitrack_frame,
                                                    table_rotation_optitrack_frame)
        table_pose_world_frame = table_rigid_body.pose.transform(table_center_pose_rb_frame)

        marker_positions_table_frame = table_center_pose_rb_frame.transform(marker_positions, inverse=True)
        table_extents_x = np.mean(
            marker_positions_table_frame[[right_bottom_marker, right_top_marker], 0] -
            marker_positions_table_frame[[left_bottom_marker, left_top_marker], 0]) + barrier_width

        # The upper barrier is variable
        upper_boundary_marker_pos = np.array([m.pos for m in table_upper_boundary_rigid_body.description.markers])
        upper_boundary_table_frame = table_pose_world_frame.transform(
            table_upper_boundary_rigid_body.pose, inverse=True)
        ub_marker_pos_table_frame = upper_boundary_table_frame.transform(upper_boundary_marker_pos)
        # Remove the last one as it is on the side of the boundary
        ub_marker_y_coords = np.sort(ub_marker_pos_table_frame[:, 1])[:-1]
        ub_mean_y_coord = np.mean(ub_marker_y_coords)

        table_lower_lim_y = np.mean(marker_positions_table_frame[[right_bottom_marker, left_bottom_marker], 1])
        table_upper_lim_y = np.mean(marker_positions_table_frame[[right_top_marker, left_top_marker], 1])
        table_extents_y = np.mean(ub_mean_y_coord - table_lower_lim_y) + barrier_width
        max_table_extents_y = np.mean(table_upper_lim_y - table_lower_lim_y) + barrier_width

        table_center_offset_y = - (max_table_extents_y - table_extents_y) / 2
        offset_center_center_frame = Transformation([0, table_center_offset_y, 0])
        table_center_pose_rb_frame = table_center_pose_rb_frame.transform(offset_center_center_frame)
        table_extents = np.array([table_extents_x, table_extents_y])
        return table_center_pose_rb_frame, table_extents

    def __find_ball_candidates(self, markers: Tuple[LabeledMarker, ...],
                               coordinates_in_optitrack_frame: bool = True) -> np.ndarray:
        marker_candidates = [m for m in markers if not m.has_model]
        if len(marker_candidates) == 0:
            return np.empty((0, 2))
        marker_positions = np.array([m.pos for m in marker_candidates])
        if coordinates_in_optitrack_frame:
            marker_positions_world_frame = self.environment.optitrack.optitrack_to_world_transformation.transform(
                marker_positions)
        else:
            marker_positions_world_frame = marker_positions
        marker_positions_table_frame = self._table_top_center_pose.inv.transform(marker_positions_world_frame)
        pos = marker_positions_table_frame
        t_lim_pos = self.table_extents + 0.2
        t_lim_neg = -self.table_extents - 0.2
        t_lim_neg[1] -= self.__pickup_location_offset_y
        marker_correct_height = np.logical_and(pos[:, 2] >= 0, pos[:, 2] <= 0.04)
        marker_within_table_extents = np.logical_and(
            np.all(pos[:, :2] >= t_lim_neg, axis=-1), np.all(pos[:, :2] <= t_lim_pos, axis=-1))
        marker_is_valid_candidate = np.logical_and(marker_correct_height, marker_within_table_extents)
        return pos[np.where(marker_is_valid_candidate)[0], :2]

    def __on_optitrack_frame_received(self, frame: DataFrame):
        self.__optitrack_frame_queue.put(frame)

    def __refresh_table_pose(self):
        new_table_pose_world_frame = self.__table_rigid_body.pose.transform(self.__table_center_pose_rb_frame)
        if self._table_top_center_initial_pose is not None:
            table_movement = self._table_top_center_initial_pose.transform(new_table_pose_world_frame, inverse=True)
            lin_movement = np.linalg.norm(table_movement.translation)
            ang_movement_deg = table_movement.angle * 180 / np.pi
            assert lin_movement < 0.05 and ang_movement_deg < 3.0, \
                "Table moved too much ({:0.2f}mm and {:0.4f}°)".format(lin_movement * 1000, ang_movement_deg)
        else:
            self._table_top_center_initial_pose = new_table_pose_world_frame
        if self._table_top_center_pose is not None:
            table_movement = self._table_top_center_pose.transform(new_table_pose_world_frame, inverse=True)
            lin_movement = np.linalg.norm(table_movement.translation)
            ang_movement_deg = table_movement.angle * 180 / np.pi
            assert lin_movement < 0.01 and ang_movement_deg < 1.0, \
                "Table moved too much in one episode ({:0.2f}mm and {:0.4f}°)".format(
                    lin_movement * 1000, ang_movement_deg)

        self._table_top_center_pose = new_table_pose_world_frame

    def __refresh_ball_state(self):
        while not self.__optitrack_frame_queue.empty():
            frame = self.__optitrack_frame_queue.get()
            position_candidates = self.__find_ball_candidates(frame.labeled_markers)
            if len(position_candidates) == 1:
                expected_state, _ = self.__ball_tracker.predict_state_at(frame.suffix.timestamp)
                expected_pos = expected_state[:2]
                deviation = np.linalg.norm(expected_pos - position_candidates[0])
                if self.__check_ball_state and deviation > 0.2:
                    raise BallTrackingException(
                        "Ball position deviated too much ({:0.2f}mm) from expected position.".format(deviation * 1000))
                self.__ball_tracker.add_observation(position_candidates[0], frame.suffix.timestamp)
                self.__last_ball_sighting = frame.suffix.timestamp
            elif self.__optitrack_frame_queue.empty():
                ball_pos_lost_since = frame.suffix.timestamp - self.__last_ball_sighting
                if self.__check_ball_state and ball_pos_lost_since > 0.2:
                    raise BallTrackingException(
                        "Lost ball position for {:0.2f}ms.".format(ball_pos_lost_since * 1000))
        self._ball_2d_position_table_frame, self._ball_2d_velocity_table_frame = self.__ball_tracker.mean_pos_vel

    def _step_task(self) -> Tuple[bool, Dict]:
        self.__intermediate_step()
        return super(SisyphusTaskReal, self)._step_task()

    def __intermediate_step(self):
        if self.__use_ball:
            self.__refresh_ball_state()
        self.refresh_visualization()
        if self.__enable_sanity_check:
            self.__check_robot_sanity(critical_only=not self.__during_episode)

    def __critical_safety_assert(self, condition: bool, text: str):
        if not condition:
            self._robot.arm.protective_stop()
            raise CriticalSafetyException("Critical safety violation: " + text)

    def __safety_assert(self, condition: bool, text: str):
        if not condition:
            self._robot.arm.stop()
            raise SafetyException("Safety violation: " + text)

    def __safety_assert_within(
            self, value: float, value_name: str, critical_lower_lim: float = -np.inf,
            critical_upper_lim: float = np.inf, lower_lim: float = -np.inf, upper_lim: float = np.inf,
            value_display_unit: str = "", value_display_scale: float = 1.0, skip_non_critical_check: bool = False):
        limits_str = "{:0.2f} {}, {:0.2f} {}.".format(
            lower_lim * value_display_scale, value_display_unit, upper_lim * value_display_scale, value_display_unit)
        critical_limits_str = "{:0.2f} {}, {:0.2f} {}.".format(
            critical_lower_lim * value_display_scale, value_display_unit, critical_upper_lim * value_display_scale,
            value_display_unit)
        text_template = "{} ({:0.2f} {}) off limits [{{}}].".format(
            value_name, value * value_display_scale, value_display_unit)
        self.__critical_safety_assert(
            critical_lower_lim <= value <= critical_upper_lim,
            text_template.format(critical_limits_str))
        if not skip_non_critical_check:
            self.__safety_assert(lower_lim <= value <= upper_lim, text_template.format(limits_str))

    def __check_robot_sanity(self, critical_only: bool = False):
        self.__critical_safety_assert(self.__optitrack_gripper.last_update_time is not None, "Optitrack not active.")
        self.__safety_assert_within(
            time.time() - self.__optitrack_gripper.last_update_time, "Time since last Optitrack gripper update",
            critical_upper_lim=1.0, value_display_unit="ms", value_display_scale=1000,
            skip_non_critical_check=critical_only)
        self.__critical_safety_assert(self.__optitrack_gripper.tracking_valid, "Lost track of gripper in optitrack.")
        marker_pose_tcp_frame = self._robot.arm.hand_eye_calibration_result.marker_pose_tcp_frame
        ot_gripper_pose = self.__optitrack_gripper.pose
        ot_tcp_pose_world_frame = ot_gripper_pose * marker_pose_tcp_frame.inv
        ot_tcp_pose_table_frame = self.table_top_center_pose.inv * ot_tcp_pose_world_frame
        ot_finger_pose_table_frame = ot_tcp_pose_table_frame * self.rotated_fingertip_pose_tcp_frame
        ot_finger_pos = ot_finger_pose_table_frame.translation
        linear_slack = 0.01
        linear_slack_critical = 0.02
        lim = self.table_top_finger_limits
        s_lim = lim + np.array([linear_slack, linear_slack])
        c_lim = lim + np.array([linear_slack_critical, linear_slack_critical])
        self.__safety_assert_within(
            ot_finger_pos[0], "Finger X position", critical_lower_lim=-c_lim[0],
            critical_upper_lim=c_lim[0], lower_lim=-s_lim[0], upper_lim=s_lim[0], value_display_unit="mm",
            value_display_scale=1000.0, skip_non_critical_check=critical_only)
        y_lower_lim = -self._table_extents[1] / 2 + self.__pickup_location_offset_y
        self.__safety_assert_within(
            ot_finger_pos[1], "Finger Y position", critical_lower_lim=y_lower_lim - linear_slack,
            critical_upper_lim=c_lim[1], lower_lim=y_lower_lim - linear_slack_critical, upper_lim=s_lim[1],
            value_display_unit="mm", value_display_scale=1000.0, skip_non_critical_check=critical_only)
        self.__safety_assert_within(
            ot_finger_pos[2], "Finger Z position", critical_lower_lim=0.005, critical_upper_lim=0.15,
            value_display_unit="mm", value_display_scale=1000.0, skip_non_critical_check=critical_only)
        z_rot_angle = np.arccos(ot_finger_pose_table_frame.rotation.apply(np.array([0, 0, 1]))[2])
        self.__safety_assert_within(
            z_rot_angle, "Gripper angle to table Z axis", critical_upper_lim=0.05, upper_lim=0.03,
            value_display_unit=" deg", value_display_scale=180 / np.pi, skip_non_critical_check=critical_only)
        self.__safety_assert_within(
            ot_finger_pose_table_frame.angle, "Gripper rotation around Z axis",
            critical_upper_lim=self.__finger_rotation_range + 0.05, upper_lim=self.__finger_rotation_range + 0.03,
            value_display_unit=" deg", value_display_scale=180 / np.pi, skip_non_critical_check=critical_only)
        if self.__use_ball:
            self.__safety_assert_within(
                self.ball_2d_position_table_frame[0], "Ball X position", critical_upper_lim=self.table_extents[0] / 2,
                critical_lower_lim=-self.table_extents[0] / 2, value_display_unit="mm", value_display_scale=1000.0,
                skip_non_critical_check=critical_only)
            self.__safety_assert_within(
                self.ball_2d_position_table_frame[1], "Ball Y position", critical_upper_lim=self.table_extents[1] / 2,
                critical_lower_lim=-self.table_extents[1] / 2 + self.__pickup_location_offset_y - 0.03,
                value_display_unit="mm", value_display_scale=1000.0,
                skip_non_critical_check=critical_only)
        # Compare Optitrack tcp pose to robot tcp pose
        pose_error = self._robot.gripper.pose.inv * ot_tcp_pose_world_frame
        if np.linalg.norm(pose_error.translation) > 0.01:
            self.__gripper_pos_off_since += 1
        else:
            self.__gripper_pos_off_since = 0
        if pose_error.angle > 3.0 * np.pi / 180:
            self.__gripper_ang_off_since += 1
        else:
            self.__gripper_ang_off_since = 0
        if self.__gripper_pos_off_since >= 10:
            self.__safety_assert_within(
                np.linalg.norm(pose_error.translation), "Optitrack-robot linear pose error",
                critical_upper_lim=0.01 * np.inf, value_display_unit=" mm", value_display_scale=1000.0)
        if self.__gripper_ang_off_since >= 10:
            self.__safety_assert_within(
                pose_error.angle, "Optitrack-robot angular pose error", critical_upper_lim=3.0 * np.pi / 180 * np.inf,
                value_display_unit=" deg", value_display_scale=180 / np.pi)
        if not self.__during_episode and self.__resetting_start_time is not None:
            self.__critical_safety_assert(time.time() - self.__resetting_start_time < 120.0,
                                          "Resetting took longer than 2 minutes. Aborting.")

    def __move_fingertip_towards(
            self, target_fingertip_pose_table_frame: Transformation, linear_velocity: float = 0.25,
            angular_velocity: Optional[float] = None, linear_target_acceleration: Optional[float] = None,
            angular_target_acceleration: Optional[float] = None):
        target_tcp_pose_table_frame = target_fingertip_pose_table_frame.transform(
            self.rotated_fingertip_pose_tcp_frame.inv)
        target_tcp_pose_world_frame = self._table_top_center_pose.transform(target_tcp_pose_table_frame)
        self._robot.arm.move_towards_pose_linear(
            target_tcp_pose_world_frame, linear_velocity=linear_velocity, angular_velocity=angular_velocity,
            linear_target_acceleration=linear_target_acceleration,
            angular_target_acceleration=angular_target_acceleration)
        return target_tcp_pose_world_frame

    def __move_fingertip_to(self, target_fingertip_pose_table_frame: Transformation, velocity: float = 0.25):
        target_tcp_pose_world_frame = self.__move_fingertip_towards(target_fingertip_pose_table_frame, velocity)
        done = False
        last_robot_move_time = time.time()
        while not done:
            self.environment.step()
            self.__intermediate_step()
            pose_error = self._robot.gripper.pose.inv * target_tcp_pose_world_frame
            lin_error = np.linalg.norm(pose_error.translation)
            ang_error = pose_error.angle
            done = lin_error < 0.0005 and ang_error < 0.001
            if np.max(self._robot.gripper.velocity) > 1e-4:
                last_robot_move_time = time.time()
                self.__robot_has_moved = True
            if not done and time.time() - last_robot_move_time > 0.5:
                # Reconnect if the robot does not start moving (happens sometimes at the beginning)
                if not self.__robot_has_moved:
                    logger.warning("Robot is not moving, reconnecting control interface...")
                    self._robot.arm.reconnect(force_disconnect_control=True)
                    self.__robot_has_moved = False
                self.__move_fingertip_towards(target_fingertip_pose_table_frame, velocity)
                last_robot_move_time = time.time()
            time.sleep(0.02)

    def _reset_task(self):
        logger.info("Resetting...")
        self.environment.synchronized_mode = False
        ok = False
        while not ok:
            try:
                self.__refresh_table_pose()
                logger.info("New table pose is {}.".format(self._table_top_center_pose))
                if self.__use_ball:
                    self.__check_ball_state = False
                    target_pos = self.finger_pose_table_frame.translation.copy()
                    target_pos[2] = self.__finger_transfer_height
                    self.__move_fingertip_to(Transformation(target_pos))
                    pickup_pos_high = [0.0, -self._table_extents[1] / 2 + self.__pickup_location_offset_y,
                                       self.__finger_transfer_height]
                    self.__move_fingertip_to(Transformation(pickup_pos_high))
                    ball_stopped = False
                    while not ball_stopped:
                        time.sleep(0.05)  # Do not DDOS the robot
                        self.environment.step()
                        self.__intermediate_step()
                        ball_stopped = np.linalg.norm(self.ball_2d_velocity_table_frame) < 0.001
                    ball_pos_x = self.ball_2d_position_table_frame[0]
                    pickup_pos_y = -self._table_extents[1] / 2 + self.__pickup_location_offset_y
                    attempts = 0
                    while abs(ball_pos_x) > 0.007 and attempts <= 10:
                        # Ball is stuck on the boundary, recover it
                        logger.info("Ball did not return to the collection point. Attempt {} of recovering it.".format(
                            attempts + 1))
                        ball_pos_sign = np.sign(ball_pos_x)
                        self.__move_fingertip_to(Transformation([0.0, pickup_pos_y, self.fingertip_distance_to_table]),
                                                 velocity=0.1)
                        self.__move_fingertip_to(
                            Transformation([0.0, -self._table_extents[1] / 2 + self.__boundary_begins_offset_y,
                                            self.fingertip_distance_to_table]), velocity=0.1)
                        stop_pos = [ball_pos_sign * (self.table_top_finger_limits[0] + 0.015),
                                    -self._table_extents[1] / 2, self.fingertip_distance_to_table]
                        self.__move_fingertip_to(Transformation(stop_pos))
                        lift_pos = stop_pos.copy()
                        lift_pos[2] = self.__finger_transfer_height
                        self.__move_fingertip_to(Transformation(lift_pos))
                        self.__move_fingertip_to(Transformation(pickup_pos_high))
                        self.environment.step()
                        self.__intermediate_step()
                        while np.linalg.norm(self.ball_2d_velocity_table_frame) >= 0.001:
                            time.sleep(0.02)  # Do not DDOS the robot
                            self.environment.step()
                            self.__intermediate_step()
                        ball_pos_x = self.ball_2d_position_table_frame[0]
                        attempts += 1
                    pickup_pos = [ball_pos_x, pickup_pos_y, self.fingertip_distance_to_table]
                    self.__move_fingertip_to(Transformation(pickup_pos))
                    initial_ball_pos_x = ball_pos_x
                    done = False
                    self.__check_ball_state = True
                    while not done:
                        finger_pos_x = self.finger_pose_table_frame.translation[0]
                        ball_pos_x = self.ball_2d_position_table_frame[0]
                        ball_vel_x = self.ball_2d_velocity_table_frame[0]
                        if self.finger_pose_table_frame.translation[1] < -0.37:
                            finger_x_pos = self.__pickup_correction_x
                        else:
                            finger_x_pos = np.clip(ball_pos_x, -0.04, 0.04)
                        target_fingertip_pos_table_frame = \
                            [finger_x_pos, -self._table_extents[1] / 2 + self.__initial_gripper_pos_offset_y,
                             self.fingertip_distance_to_table]
                        # target_fingertip_rotation_z = np.clip(
                        #     0.0 * (ball_pos_x - finger_pos_x) + 20.0 * ball_vel_x,
                        #     -self.__finger_rotation_range, self.__finger_rotation_range)
                        target_fingertip_rotation_z = 0.0
                        target_fingertip_pose_table_frame = Transformation.from_pos_euler(
                            target_fingertip_pos_table_frame, [0, 0, target_fingertip_rotation_z])
                        # Reduce linear acceleration to prevent jittering
                        self.__move_fingertip_towards(target_fingertip_pose_table_frame, 0.1, 0.8,
                                                      linear_target_acceleration=0.5)
                        self.environment.step()
                        self.__intermediate_step()
                        pos_error_y = abs(
                            self.finger_pose_table_frame.translation[1] - target_fingertip_pos_table_frame[1])
                        ball_stopped = np.linalg.norm(self.ball_2d_velocity_table_frame) < 0.001
                        ball_lost = self.ball_2d_position_table_frame[1] < self.finger_pose_table_frame.translation[1]
                        if ball_lost:
                            logger.info("Lost ball. Retrying...")
                        done = pos_error_y < 0.005 and ball_stopped or ball_lost
                        ok = done and not ball_lost
                        time.sleep(0.02)  # Do not DDOS the robot
                    self.__refresh_ball_state()
                else:
                    # No need to move the finger up in this scenario
                    self.environment.step()
                    finger_pos_z = self.finger_pose_table_frame.translation[2]
                    self.__move_fingertip_to(
                        Transformation(
                            [0.0, -self._table_extents[1] / 2 + self.__initial_gripper_pos_offset_y, finger_pos_z]))
                    self.__move_fingertip_to(
                        Transformation(
                            [0.0, -self._table_extents[1] / 2 + self.__initial_gripper_pos_offset_y,
                             self.fingertip_distance_to_table]))
                    ok = True
                self.refresh_visualization()
            except (BallTrackingException, SafetyException) as ex:
                logger.warning("Caught exception during reset: {}".format(ex))
                ok = False
        self.environment.synchronized_mode = True
        logger.info("Resetting done.")

    def __log_robot_error(self, desc: str, exception_text: str):
        text = "\n".join(
            ["[{}] {}".format(time.time(), desc), exception_text, "#" * 20 + " ACTION LOG " + "#" * 20,
             self._robot.arm.get_action_protocol_str()])
        if self.__log_dir is None:
            logger.error(text)
        else:
            self.__log_dir.mkdir(exist_ok=True, parents=True)
            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = "error_{}.log".format(time_str)
            i = 0
            while (self.__log_dir / filename).exists():
                i += 1
                filename = "error_{}_{}.log".format(time_str, i)
            path = self.__log_dir / filename
            with path.open("w") as f:
                f.write(text)
            logger.info("Logged action protocol to {}.".format(path))

    def reset(self) -> Dict[str, np.ndarray]:
        self.__resetting_start_time = time.time()
        self.__during_episode = False
        try:
            ur_reconnect_attempts = 0
            gripper_reconnect_attempts = 0
            while True:
                try:
                    return super(SisyphusTaskReal, self).reset()
                except UR10ConnectionLostException:
                    logger.warning("Lost connection to UR10 during reset.")
                    logger.warning(traceback.format_exc())
                    self.__log_robot_error("Lost connection to UR10 during reset.", traceback.format_exc())
                    if ur_reconnect_attempts >= 10:
                        logger.error("Failed to reestablish UR10 connection.")
                        raise
                    logger.info("Attempting to reconnect...")
                    time.sleep(0.5)
                    self._robot.arm.reconnect()
                    ur_reconnect_attempts += 1
                except (DynamixelCommunicationError, DynamixelPacketError):
                    logger.warning("Lost connection to gripper during reset.")
                    logger.warning(traceback.format_exc())
                    if gripper_reconnect_attempts >= 10:
                        logger.error("Failed to reestablish RHP12RN connection.")
                        raise
                    logger.info("Attempting to reconnect...")
                    time.sleep(0.5)
                    self._robot.gripper.reconnect()
                    ur_reconnect_attempts += 1
                except KeyboardInterrupt:
                    raise
                except:
                    logger.error("An unknown error occurred during reset.")
                    logger.error(traceback.format_exc())
                    self.__log_robot_error("An unknown error occurred during reset.", traceback.format_exc())
                    raise
        finally:
            self.__during_episode = True

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        try:
            return super(SisyphusTaskReal, self).step(action)
        except UR10ConnectionLostException as ex:
            logger.warning("Lost connection to UR10 during episode.")
            logger.warning(traceback.format_exc())
            self.__log_robot_error("Lost connection to UR10 during episode.", traceback.format_exc())
            self._robot.arm.reconnect()
            raise WrappedEpisodeInvalidException(ex)
        except (DynamixelCommunicationError, DynamixelPacketError) as ex:
            logger.warning("Lost connection to gripper during episode.")
            logger.warning(traceback.format_exc())
            self._robot.gripper.reconnect()
            raise WrappedEpisodeInvalidException(ex)
        except KeyboardInterrupt:
            raise
        except:
            logger.error("An unknown error occurred during episode.")
            logger.error(traceback.format_exc())
            self.__log_robot_error("An unknown error occurred during episode.", traceback.format_exc())
            raise

    def close(self):
        if self.__pybullet is not None:
            self.__pybullet.disconnect()
        super(SisyphusTaskReal, self).close()

    def terminate_episode(self):
        try:
            super(SisyphusTaskReal, self).terminate_episode()
        except UR10ConnectionLostException:
            logger.warning("Lost connection to UR10 during episode termination.")
            self._robot.arm.reconnect()
        except (DynamixelCommunicationError, DynamixelPacketError):
            logger.warning("Lost connection to gripper during episode termination.")
            self._robot.gripper.reconnect()
