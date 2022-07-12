from pathlib import Path
from typing import Optional, Sequence

import gym

import numpy as np


class BaseEnv(gym.Env):
    def __init__(self, state_repr_lower_bounds: Sequence[float] = (-1.0, -1.0),
                 state_repr_upper_bounds: Sequence[float] = (-1.0, -1.0),
                 min_reward: Optional[float] = None, max_reward: Optional[float] = None):
        self.state_repr_upper_bounds = np.array(state_repr_upper_bounds)
        self.state_repr_lower_bounds = np.array(state_repr_lower_bounds)
        self.min_reward = min_reward
        self.max_reward = max_reward
        super(BaseEnv, self).__init__()

    def start_video_log(self, path: Path):
        pass

    def stop_video_log(self):
        pass

    def standby(self):
        pass

    def wakeup(self):
        pass

    def reset_to_state(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
