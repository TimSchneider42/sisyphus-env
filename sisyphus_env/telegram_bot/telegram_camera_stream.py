from typing import Tuple

import cv2
import numpy as np
from pytgcalls.utils import VideoInfo

from .camera import Camera, resize_image


class TelegramCameraStream:
    def __init__(self, camera: Camera, video_output_size: Tuple[int, int] = (1280, 720)):
        self.__camera = camera
        self.__video_output_size = video_output_size

    def start(self):
        pass

    def stop(self):
        pass

    def image_to_bytes(self, img: np.ndarray) -> bytes:
        return resize_image(img, self.__video_output_size).tobytes()

    def read(self):
        return self.image_to_bytes(self.__camera.get_current_frame())

    def get_video_info(self) -> VideoInfo:
        return VideoInfo(*self.__video_output_size, self.__camera.framerate)

    @property
    def is_running(self):
        return True
