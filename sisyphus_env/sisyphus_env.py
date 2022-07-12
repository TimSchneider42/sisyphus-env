from datetime import datetime
import glob
import time
from pathlib import Path
import json
from typing import Optional, Tuple, Sequence, Literal, TYPE_CHECKING

import numpy as np

from robot_gym.core.base_task import EpisodeInvalidException
from robot_gym.core.controllers import ConstantActionController, JointPositionController
from robot_gym.core.rewards import ActionReward
from robot_gym.core.wrappers import FlattenWrapper
from .logger import logger
from .endeffector_velocity_controller import SisyphusEndEffectorVelocityController
from .sisyphus_reward import SisyphusReward
from .sisyphus_task import SisyphusTask
from .sisyphus_task_sim_realistic import SisyphusTaskSimRealistic
from .sisyphus_task_sim_simple import SisyphusTaskSimSimple
from .ball_position_sensor import BallPositionSensor
from .ball_velocity_sensor import BallVelocitySensor
from .fingertip_pose_sensor import FingertipPose2DSensor
from .fingertip_target_velocity_sensor import FingertipTargetVelocity2DSensor
from .fingertip_velocity_sensor import FingertipVelocity2DSensor
from .reaching_reward import ReachingReward

import pybullet

from robot_gym.environment import Robot
from transformation import Transformation
from .base_env import BaseEnv

from robot_gym.environment.pybullet import PybulletEnvironment
from .rectangle import Rectangle

if TYPE_CHECKING:
    from telegram_bot import TelegramBot
    from pyrogram.types import Message

try:
    from .sisyphus_task_real import SisyphusTaskReal
    from rhp12rn import RHP12RNAConnector
    from robot_gym.environment.real import RealEnvironment, Optitrack, RHP12RNGripper, UR10RobotArm, \
        HandEyeCalibrationResult
except ImportError:
    SisyphusTaskReal = RHP12RNAConnector = RealEnvironment = Optitrack = RHP12RNGripper = UR10RobotArm = \
        HandEyeCalibrationResult = None


class SisyphusEnv(BaseEnv):
    def __init__(self, use_ball: bool = True, allow_rotation: bool = False, headless: bool = False,
                 action_weight: float = 0.0, gripper_width: float = 0.02,
                 ball_friction: float = 1.0, ball_mass: float = 0.02, control_lin_accel: bool = False,
                 max_lin_accel: float = 1.0, holes: Sequence[Rectangle] = (),
                 platform: Literal["real", "sim", "sim_rt"] = "sim", dense_reward: bool = False,
                 power_off_robot_on_close: bool = True, shutdown_robot_on_close: bool = False,
                 use_telegram_bot: bool = False, done_on_border_contact: bool = False, done_on_ball_lost: bool = False,
                 time_steps: int = 50, log_dir: Optional[Path] = None, obs_act_offset: float = 0.2,
                 table_extents: Tuple[float, float] = (0.5, 0.57), sim_table_inclination_rad: float = 0.2):

        self.__controller_interval = 0.25
        self.__repeat_actions = 1
        self.__control_lin_accel = control_lin_accel
        self.__max_lin_accel = max_lin_accel
        self.__telegram_bot: Optional["TelegramBot"] = None
        self.__telegram_video_buffer_size = 40.0
        self.__telegram_video_length = 20.0
        self.__shutdown_robot_on_close = shutdown_robot_on_close
        self.__power_off_robot_on_close = power_off_robot_on_close
        self.__video_log_path = None
        self.__video_start_time = None

        root = Path(__file__).parents[3]
        if log_dir is not None:
            self.__video_error_log_dir = log_dir / "video_error_log"
        else:
            self.__video_error_log_dir = None

        self.__finger_linear_vel_lim_lower = np.array([-0.15, -0.15])
        self.__finger_linear_vel_lim_upper = np.array([0.15, 0.15])
        self.__finger_angular_vel_lim_lower = -3.0
        self.__finger_angular_vel_lim_upper = 3.0

        self.__allow_rotation = allow_rotation
        self.__use_ball = use_ball
        self.__headless = headless
        self.__gripper_width = gripper_width
        self.__ball_friction = ball_friction
        self.__ball_mass = ball_mass
        self.__holes = holes
        self.__dense_reward = dense_reward
        self.__use_telegram_bot = use_telegram_bot
        self.__done_on_border_contact = done_on_border_contact
        self.__done_on_ball_lost = done_on_ball_lost
        self.__log_dir = log_dir
        self.__obs_act_offset = obs_act_offset
        self.__table_extents = table_extents
        self.__time_steps = time_steps
        self.__platform = platform
        self.__action_weight = action_weight
        self.__sim_table_inclination_rad = sim_table_inclination_rad

        self.__max_rotation_range = 0.3
        self.__rotation_range = self.__max_rotation_range if self.__allow_rotation else 0.0

        self.__inner_env = self.__make_env()

        state_repr_lower_bounds = self.__inner_env.observation_space.low[:2]
        state_repr_upper_bounds = self.__inner_env.observation_space.high[:2]
        self.observation_space = self.__inner_env.observation_space
        self.action_space = self.__inner_env.action_space
        super(SisyphusEnv, self).__init__(
            min_reward=0, max_reward=time_steps * (1 + action_weight),
            state_repr_lower_bounds=state_repr_lower_bounds, state_repr_upper_bounds=state_repr_upper_bounds)
        if self.__platform == "real" and self.__use_telegram_bot:
            from .telegram_bot import TelegramBot
            calibration_path = root / "calibration" / "camera_params.json"
            if calibration_path.exists():
                with calibration_path.open() as f:
                    camera_params = json.load(f)
                auto_exposure = camera_params["auto_exposure"]
                exposure = camera_params["exposure"]
                gain = camera_params["gain"]
            else:
                auto_exposure = exposure = gain = None
            self.__telegram_bot = TelegramBot(
                root / "data" / "telegram", video_device="/dev/video6",
                video_buffer_size_s=self.__telegram_video_buffer_size, video_resolution=(1920, 1080),
                video_resolution_send=(640, 480), auto_exposure=auto_exposure, exposure=exposure, gain=gain)
            self.__telegram_bot.on_message_event.handlers.append(self.telegram_message_handler)
            self.__telegram_bot.start()
        else:
            self.__telegram_bot: Optional["TelegramBot"] = None
        self.telegram_send_message("Starting experiment 🔥💯🔥")

    def __make_env(self):
        arm_controller = SisyphusEndEffectorVelocityController(
            "ur10", linear_limits_lower=self.__finger_linear_vel_lim_lower,
            linear_limits_upper=self.__finger_linear_vel_lim_upper,
            angular_limit_lower=self.__finger_angular_vel_lim_lower,
            angular_limit_upper=self.__finger_angular_vel_lim_upper,
            rotation_range=self.__rotation_range, control_lin_accel=self.__control_lin_accel,
            max_lin_accel=self.__max_lin_accel, disable_rotation=not self.__allow_rotation)
        gripper_controller = ConstantActionController(
            JointPositionController("ur10", "gripper", np.zeros(1), np.ones(1)), np.array([1.1]))

        sensors = [
            FingertipPose2DSensor(rotation_range=self.__rotation_range),
            FingertipVelocity2DSensor(
                "ur10", self.__finger_linear_vel_lim_lower, self.__finger_linear_vel_lim_upper,
                self.__finger_angular_vel_lim_lower, self.__finger_angular_vel_lim_upper,
                sense_angle=self.__allow_rotation),
        ]
        if self.__use_ball:
            sensors += [BallPositionSensor(), BallVelocitySensor()]
        if self.__platform in ["real", "sim_rt"]:
            sensors.append(
                FingertipTargetVelocity2DSensor(
                    self.__finger_linear_vel_lim_lower, self.__finger_linear_vel_lim_upper,
                    self.__finger_angular_vel_lim_lower, self.__finger_angular_vel_lim_upper,
                    sense_angle=self.__allow_rotation))

        placing_reward = SisyphusReward(dense=self.__dense_reward) if self.__use_ball else ReachingReward(
            dense=self.__dense_reward)
        reward = placing_reward + self.__action_weight * ActionReward()

        assert self.__platform in ["sim", "sim_rt", "real"]

        if self.__platform == "real" and RealEnvironment is None:
            logger.warning("Robot drivers could not be loaded. Using sim_rt environment instead of real environment.")
            self.__platform = "sim_rt"

        if self.__platform in ["sim", "sim_rt"]:
            if self.__platform == "sim":
                task = SisyphusTaskSimSimple(
                    [arm_controller, gripper_controller], sensors, reward, time_step=self.__controller_interval,
                    use_ball=self.__use_ball, gripper_width=self.__gripper_width, ball_friction=self.__ball_friction,
                    holes=self.__holes, table_inclination_rad=self.__sim_table_inclination_rad,
                    ball_mass=self.__ball_mass, table_extents=self.__table_extents, time_limit_steps=self.__time_steps)
                sim_env = PybulletEnvironment(headless=self.__headless)
            else:
                task = SisyphusTaskSimRealistic(
                    [arm_controller, gripper_controller], sensors, reward, time_step=self.__controller_interval,
                    use_ball=self.__use_ball, gripper_width=self.__gripper_width, ball_friction=self.__ball_friction,
                    ball_mass=self.__ball_mass, table_extents=self.__table_extents, time_limit_steps=self.__time_steps,
                    obs_to_act_time=self.__obs_act_offset)
                sim_env = PybulletEnvironment(headless=self.__headless, simulator_step_s=0.008)
            task.initialize(sim_env)
            ur10 = sim_env.robots["ur10"].arm
            ur10.linear_acceleration_lin = 1.0
            ur10.linear_acceleration_ang = 14.0
            sim_env.physics_client.call(
                pybullet.resetDebugVisualizerCamera, cameraDistance=0.7, cameraYaw=180.0, cameraPitch=-60.0,
                cameraTargetPosition=np.array(task.table_top_center_pose.translation))
            sim_env.physics_client.call(
                pybullet.configureDebugVisualizer, pybullet.COV_ENABLE_GUI, 0)
        else:
            assert RealEnvironment is not None, "Robot drivers could not be loaded."
            root = Path(__file__).parents[3]
            with (root / "calibration" / "hand_eye_calibration_result.json").open() as f:
                hec_calibration = HandEyeCalibrationResult.from_dict(json.load(f))
            usb_devices = glob.glob("/dev/ttyUSB*")
            gripper = RHP12RNGripper(
                RHP12RNAConnector(usb_devices[0], baud_rate=2000000, dynamixel_id=1), async_actions=True)
            arm = UR10RobotArm("192.168.1.101", ur_cap_port=50002, optitrack_tcp_name="gripper",
                               fixed_hand_eye_calibration=hec_calibration, angular_target_acceleration=14.0,
                               linear_target_acceleration=1.0, watchdog_timeout_execution=1.0)
            gripper.attach_to(arm)
            robot = Robot(arm, gripper, "ur10")
            optitrack = Optitrack(
                server_ip_address="192.168.1.35", local_ip_address="192.168.1.34", use_multicast=False,
                world_transformation=Transformation.from_pos_euler(euler_angles=[np.pi / 2, 0, 0]))

            env = RealEnvironment(optitrack, [robot], obs_act_offset_s=self.__obs_act_offset)
            task = SisyphusTaskReal(
                [arm_controller, gripper_controller], sensors, reward, time_step=self.__controller_interval,
                headless=self.__headless, finger_rotation_range=self.__max_rotation_range, use_ball=self.__use_ball,
                done_on_border_contact=self.__done_on_border_contact, done_on_ball_lost=self.__done_on_ball_lost,
                max_time_steps=self.__time_steps, log_dir=self.__log_dir)
            task.initialize(env)
        return FlattenWrapper(task)

    def telegram_send_message(self, msg: str, attach_vid: bool = False):
        if self.__telegram_bot is not None:
            self.__telegram_bot.broadcast_message(msg, self.__telegram_video_length if attach_vid else None)

    def telegram_message_handler(self, msg: "Message"):
        env = self.__inner_env.unwrapped
        assert isinstance(env, SisyphusTaskReal)
        cmd = msg.text.lower()
        if cmd in ["vid", "👀"]:
            self.telegram_send_message("", True)
        elif cmd == "ping":
            self.telegram_send_message("pong")
        elif cmd in ["stop", "🛑", "✋", "🚏"]:
            env.environment.robots["ur10"].arm.protective_stop()
            time.sleep(1.0)
            self.telegram_send_message("Here is how the protective stop turned out.", True)
        elif cmd == "shutdown":
            self.shutdown_robot()
            time.sleep(1.0)
            self.telegram_send_message("Here is how the shutdown turned out.", True)
        elif cmd == "poweroff":
            self.power_off_robot()
        else:
            self.telegram_send_message("Unkown command \"{}\"".format(cmd))

    def step(self, action: np.ndarray):
        try:
            reward = 0
            s, r, d, i = [None] * 4
            for ai in range(self.__repeat_actions):
                s, r, d, i = self.__inner_env.step(action)
                if ai == self.__repeat_actions - 1:
                    reward = r
            i["state_repr"] = s[:2]  # This is the position of the ball
            return s, reward, d, i
        except EpisodeInvalidException:
            raise
        except Exception as ex:
            self.log_experiment_failed(ex)
            try:
                self.__inner_env.unwrapped.terminate_episode()
            except:
                pass
            raise

    def reset(self):
        try:
            output = self.__inner_env.reset()
            return output
        except Exception as ex:
            self.log_experiment_failed(ex)
            try:
                self.__inner_env.unwrapped.terminate_episode()
            except:
                pass
            raise

    def log_experiment_failed(self, ex: Exception):
        if SisyphusTaskReal is not None and isinstance(self.__inner_env.unwrapped, SisyphusTaskReal):
            self.telegram_send_message("Experiment failed with: {} 😢".format(ex))
            self.telegram_send_message("Events leading up to the catastrophe", True)
            if self.__telegram_bot is not None and self.__video_error_log_dir is not None:
                self.__video_error_log_dir.mkdir(exist_ok=True)
                video_name = "error_log_{}.mp4".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                self.__telegram_bot.save_video_sequence(self.__video_error_log_dir / video_name)

    def reset_to_state(self, state: np.ndarray) -> np.ndarray:
        state_unpacked = self.__inner_env.observation_flattener.unpack_dict(state)
        assert "ball_position" in state_unpacked, "Ball position has to be known for reset_to_state"
        task: SisyphusTask = self.__inner_env.unwrapped
        te = task.table_top_accessible_extents
        task.initial_ball_pos_override = state_unpacked["ball_position"] * te / 2
        return self.reset()

    def render(self, mode="human"):
        return self.__inner_env.render(mode)

    def shutdown_robot(self):
        env = self.__inner_env.unwrapped
        if SisyphusTaskReal is not None and isinstance(env, SisyphusTaskReal):
            self.telegram_send_message("Shutting down robot...")
            env.environment.robots["ur10"].arm.shutdown_robot()

    def power_off_robot(self):
        env = self.__inner_env.unwrapped
        if SisyphusTaskReal is not None and isinstance(env, SisyphusTaskReal):
            self.telegram_send_message("Powering down robot...")
            env.environment.robots["ur10"].arm.power_off()

    def close(self):
        return self.__close("Experiment terminated", shutdown=self.__shutdown_robot_on_close)

    def __close(self, msg: str, stop_telegram_bot: bool = True, shutdown: bool = False):
        if self.__inner_env is not None:
            ret_val = self.__inner_env.close()
        else:
            ret_val = None
        if self.__telegram_bot is not None:
            self.telegram_send_message(msg)
            if stop_telegram_bot:
                self.__telegram_bot.stop()
        if shutdown:
            self.shutdown_robot()
        elif self.__power_off_robot_on_close:
            self.power_off_robot()
        return ret_val

    def standby(self):
        self.__close("Experiment paused.", stop_telegram_bot=False)
        self.__inner_env = None

    def wakeup(self):
        assert self.__inner_env is None
        self.__inner_env = self.__make_env()
        self.telegram_send_message("Resuming experiment.")

    def seed(self, seed=None):
        return self.__inner_env.seed(seed)

    @property
    def unwrapped(self) -> SisyphusTask:
        return self.__inner_env.unwrapped

    def start_video_log(self, path: Path):
        self.__video_log_path = path
        self.__video_start_time = time.time()

    def stop_video_log(self):
        length = min(time.time() - self.__video_start_time + 0.1, self.__telegram_video_buffer_size)
        if self.__telegram_bot is not None:
            self.__telegram_bot.save_video_sequence(str(self.__video_log_path), video_length=length)
        self.__video_log_path = None
        self.__video_start_time = None

    @property
    def repeat_actions(self) -> int:
        return self.__repeat_actions

    @property
    def control_interval(self) -> float:
        return self.__controller_interval

    @property
    def inner_env(self) -> FlattenWrapper:
        return self.__inner_env

    @property
    def rotation_range(self) -> float:
        return self.__rotation_range
