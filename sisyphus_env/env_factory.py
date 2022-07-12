from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .rectangle import Rectangle
from .sisyphus_env import SisyphusEnv


def mk_sisyphus(
        headless: bool = False, log_dir: Optional[Path] = None, allow_finger_rotation: bool = True, ball_mass: int = 2,
        gripper_width: float = 0.02, ball_friction: float = 0.3, use_holes: bool = False, time_step_limit: int = 30,
        table_inclination_rad: float = 0.2, dense_reward: bool = False,
        done_on_border_contact: bool = False, table_extents: Tuple[int, int] = (0.5, 0.57),
        simulate_real_time: bool = False, rt_obs_act_offset: float = 0.2) -> SisyphusEnv:
    holes = [
        Rectangle(np.array([-0.25, 0.3]), np.array([0.25, 0.5])),
        Rectangle(np.array([-1.0, -0.2]), np.array([-0.2, 0.0])),
        Rectangle(np.array([-1.0, 0.0]), np.array([-0.6, 1.0])),
        Rectangle(np.array([0.2, -0.2]), np.array([1.0, 0.0])),
        Rectangle(np.array([0.6, 0.0]), np.array([1.0, 1.0]))
    ]

    env = SisyphusEnv(
        headless=headless, allow_rotation=allow_finger_rotation, action_weight=0.001, gripper_width=gripper_width,
        ball_friction=ball_friction, holes=holes if use_holes else [], ball_mass=ball_mass,
        log_dir=log_dir, time_steps=time_step_limit, sim_table_inclination_rad=table_inclination_rad,
        dense_reward=dense_reward, done_on_border_contact=done_on_border_contact, table_extents=table_extents,
        platform="simrt" if simulate_real_time else "sim", obs_act_offset=rt_obs_act_offset)
    env.reset()
    return env


def mk_sisyphus_real(
        headless: bool = False, log_dir: Optional[Path] = None, allow_finger_rotation: bool = True,
        time_step_limit: int = 30, dense_reward: bool = False, done_on_border_contact: bool = False,
        obs_act_offset: float = 0.2, use_telegram_bot: bool = False,
        shutdown_robot_on_close: bool = False) -> SisyphusEnv:
    return SisyphusEnv(
        action_weight=0.001, platform="real", allow_rotation=allow_finger_rotation,
        headless=headless, dense_reward=dense_reward, use_telegram_bot=use_telegram_bot,
        done_on_border_contact=done_on_border_contact, shutdown_robot_on_close=shutdown_robot_on_close,
        time_steps=time_step_limit, log_dir=log_dir,
        obs_act_offset=obs_act_offset)
