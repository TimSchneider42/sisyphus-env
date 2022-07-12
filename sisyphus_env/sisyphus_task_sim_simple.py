from itertools import chain

from robot_gym.core.controllers import Controller
from robot_gym.core.rewards import Reward
from robot_gym.core.sensors import Sensor
from typing import Tuple, Iterable, Optional, Sequence, List
import numpy as np

from .rectangle import Rectangle
from transformation import Transformation
from .sisyphus_task_sim import SisyphusTaskSim


class SisyphusTaskSimSimple(SisyphusTaskSim):
    def __init__(self, controllers: Iterable[Controller],
                 sensors: Iterable[Sensor["SisyphusTaskSim"]], rewards: Reward["SisyphusTaskSim"],
                 time_step: float = 0.005, time_limit_steps: Optional[int] = None, use_ball: bool = True,
                 gripper_width: float = 0.02, ball_friction: float = 1.0, ball_mass: float = 0.02,
                 holes: Sequence[Rectangle] = (), table_extents: Tuple[float, float] = (0.5, 0.57),
                 table_inclination_rad: float = 0.2):
        """
        :param controllers:             A sequence of controller objects that define the actions on the environment
        :param sensors:                 A sequence of sensors objects that provide observations to the agent
        :param rewards:                 A sequence of rewards objects that provide rewards to the agent
        :param time_step:               The time between two controller updates (actions of the agent)
        :param time_limit_steps:        The number of steps until the episode terminates (if no other termination
                                        criterion is reached)
        """
        super(SisyphusTaskSimSimple, self).__init__(
            controllers, sensors, rewards, time_step, use_ball=use_ball, gripper_width=gripper_width,
            ball_friction=ball_friction, ball_mass=ball_mass, time_limit_steps=time_limit_steps)
        self.__table_extents = np.array(table_extents)
        self.__target_zone_height = 0.001
        self.__barrier_height = 0.03
        self.__holes = holes
        self.__hole_depth = self.ball_radius * 2
        self.__table_base_color = (0.4, 0.3, 0.3, 1.0)
        self.__surface_box_color = (0.5, 0.4, 0.4, 1.0)
        self.__boundary_color = (0.3, 0.2, 0.2, 1.0)
        self.__table_height = self.__hole_depth + 0.02
        self.__table_inclination_rad = table_inclination_rad

    def _invert_holes(self, holes: Sequence[Rectangle]) -> "List[Rectangle]":
        assert all(
            np.all(h.min_coords >= -1) and np.all(h.max_coords <= 1) and np.all(h.min_coords < h.max_coords)
            for h in holes), "Hole coordinates must be in [-1, 1]^2"
        assert all(
            not h1.intersects(h2, strict=True) for i, h1 in enumerate(holes) for h2 in
            holes[i + 1:]), "Holes must not intersect"
        cells_x = np.array(sorted({p for w in holes for p in [w.min_coords[0], w.max_coords[0]]}.union([-1, 1])))
        cells_y = np.array(sorted({p for w in holes for p in [w.min_coords[1], w.max_coords[1]]}.union([-1, 1])))
        hole_cells_x = np.array(
            [
                np.logical_and(cells_x[:-1] >= w.min_coords[0], cells_x[:-1] < w.max_coords[0])
                for w in holes
            ]).reshape((len(holes), len(cells_x) - 1))
        hole_cells_y = np.array(
            [
                np.logical_and(cells_y[:-1] >= w.min_coords[1], cells_y[:-1] < w.max_coords[1])
                for w in holes
            ]).reshape((len(holes), len(cells_y) - 1))

        cell_not_occupied = np.any(
            np.logical_and(hole_cells_x[:, np.newaxis, :], hole_cells_y[:, :, np.newaxis]), axis=0)
        cell_occupied = np.logical_not(cell_not_occupied)

        output_rects = []
        table_top_accessible_extents = np.array(self.__table_extents) - self.barrier_width * 2
        scale = table_top_accessible_extents / 2
        for y, co_arr in enumerate(cell_occupied):
            current_start_x = None
            prev_cell_occupied = False
            for x, current_cell_occupied in enumerate(chain(co_arr, [False])):
                if not current_cell_occupied and prev_cell_occupied:
                    min_coords = np.array([current_start_x, cells_y[y]]) * scale
                    max_coords = np.array([cells_x[x], cells_y[y + 1]]) * scale
                    rect = Rectangle(min_coords, max_coords)
                    output_rects.append(rect)
                if not prev_cell_occupied:
                    current_start_x = cells_x[x]
                prev_cell_occupied = current_cell_occupied
        return output_rects

    def _setup_table(self):
        # Table
        bw = self.barrier_width
        bh = self.__barrier_height
        table_top_center_pose = Transformation.from_pos_euler([0.0, 0.8, 0.7], [self.__table_inclination_rad, 0.0, 0.0])

        table_box_extents = np.concatenate([self.__table_extents, [self.__table_height]])
        if len(self.__holes) > 0:
            table_box_extents[2] -= self.__hole_depth
            table_center_top_frame = Transformation(
                translation=np.array([0.0, 0.0, -table_box_extents[2] / 2 - self.__hole_depth]))
            table_pose = table_top_center_pose.transform(table_center_top_frame)
            inverted_holes = self._invert_holes(self.__holes)
            for rect in inverted_holes:
                extents = np.concatenate([rect.max_coords - rect.min_coords, [self.__hole_depth]])
                pose_tt_center_frame = Transformation(
                    np.concatenate([(rect.max_coords + rect.min_coords) / 2, [-self.__hole_depth / 2]]))
                pose_world_frame = table_top_center_pose.transform(pose_tt_center_frame)
                self._add_box(extents, pose_world_frame, rgba_colors=self.__surface_box_color)
            self._add_box(
                [self.__table_extents[0], bw, self.__hole_depth],
                table_top_center_pose * Transformation.from_pos_euler(
                    position=[0, -(self.__table_extents[1] / 2 - bw / 2), - self.__hole_depth / 2]),
                rgba_colors=self.__table_base_color)
            self._add_box(
                [self.__table_extents[0], bw, self.__hole_depth],
                table_top_center_pose * Transformation.from_pos_euler(
                    position=[0, self.__table_extents[1] / 2 - bw / 2, - self.__hole_depth / 2]),
                rgba_colors=self.__table_base_color)
            self._add_box(
                [bw, self.__table_extents[1] - 2 * bw, self.__hole_depth],
                table_top_center_pose * Transformation.from_pos_euler(
                    position=[-(self.__table_extents[0] / 2 - bw / 2), 0, - self.__hole_depth / 2]),
                rgba_colors=self.__table_base_color)
            self._add_box(
                [bw, self.__table_extents[1] - 2 * bw, self.__hole_depth],
                table_top_center_pose * Transformation.from_pos_euler(
                    position=[self.__table_extents[0] / 2 - bw / 2, 0, - self.__hole_depth / 2]),
                rgba_colors=self.__table_base_color)
        else:
            table_top_table_frame = Transformation.from_pos_euler(position=[0, 0, self.__table_height / 2])
            table_pose = table_top_center_pose * table_top_table_frame.inv
        self._add_box(table_box_extents, table_pose, rgba_colors=self.__table_base_color)

        # Boundaries
        self._add_box(
            [self.__table_extents[0], bw, bh],
            table_top_center_pose * Transformation.from_pos_euler(
                position=[0, -(self.__table_extents[1] / 2 - bw / 2), bh / 2]),
            rgba_colors=self.__boundary_color)
        self._add_box(
            [self.__table_extents[0], bw, bh],
            table_top_center_pose * Transformation.from_pos_euler(
                position=[0, self.__table_extents[1] / 2 - bw / 2, bh / 2]),
            rgba_colors=self.__boundary_color)
        self._add_box(
            [bw, self.__table_extents[1] - 2 * bw, bh],
            table_top_center_pose * Transformation.from_pos_euler(
                position=[-(self.__table_extents[0] / 2 - bw / 2), 0, bh / 2]),
            rgba_colors=self.__boundary_color)
        self._add_box(
            [bw, self.__table_extents[1] - 2 * bw, bh],
            table_top_center_pose * Transformation.from_pos_euler(
                position=[self.__table_extents[0] / 2 - bw / 2, 0, bh / 2]),
            rgba_colors=self.__boundary_color)
        return self.__table_extents, table_top_center_pose

    def _get_initial_ball_pos(self):
        return np.random.uniform([-0.1, -0.15], [0.1, -0.15])
