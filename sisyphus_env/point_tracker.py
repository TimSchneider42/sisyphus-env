from typing import Tuple, Optional

import numpy as np


class PointTracker:
    def __init__(self, dimensions: int = 3, observation_noise: float = 0.1, position_noise_per_s: float = 0.1,
                 velocity_noise_per_s: float = 0.1):
        self.__prev_time_stamp = None
        self.__mean = np.zeros(dimensions * 2)
        self.__covariance = np.eye(dimensions * 2)
        self.__observation_covariance = np.eye(dimensions) * observation_noise
        self.__process_noise_per_s = np.diag(
            [position_noise_per_s] * dimensions + [velocity_noise_per_s] * dimensions)
        self.__observation_matrix = np.concatenate([np.eye(dimensions), np.zeros((dimensions, dimensions))], axis=1)

    def add_observation(self, position: np.ndarray, time_stamp: float):
        p_mean, p_cov = self.predict_state_at(time_stamp)
        dimensions = self.__mean.shape[0] // 2
        obs = self.__observation_matrix
        K = np.linalg.solve(obs @ p_cov.T @ obs.T + self.__observation_covariance.T, obs @ p_cov.T).T
        self.__mean = p_mean + K @ (position - obs @ p_mean)
        self.__covariance = (np.eye(dimensions * 2) - K @ obs) @ p_cov
        self.__prev_time_stamp = time_stamp

    def predict_state_at(self, time_stamp: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.__prev_time_stamp is None:
            return self.__mean, self.__covariance
        else:
            dimensions = self.__mean.shape[0] // 2
            dt = time_stamp - self.__prev_time_stamp
            transition_matrix = np.block([[np.eye(dimensions), np.eye(dimensions) * dt],
                                          [np.zeros((dimensions, dimensions)), np.eye(dimensions)]])
            process_noise = self.__process_noise_per_s * dt
            prediction_mean = transition_matrix.dot(self.__mean)
            prediction_var = transition_matrix @ self.__covariance @ transition_matrix.T + process_noise
            return prediction_mean, prediction_var

    @property
    def mean_pos_vel(self) -> Tuple[np.ndarray, np.ndarray]:
        d = self.__mean.shape[0] // 2
        mean = self.predict_state_at(self.__prev_time_stamp)[0]
        return mean[:d], mean[d:]

    @property
    def covariance(self):
        return self.predict_state_at(self.__prev_time_stamp)[1]

    @property
    def time_stamp(self) -> Optional[float]:
        return self.__prev_time_stamp
