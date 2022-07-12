from asyncio import AbstractEventLoop
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
from typing import Sequence, Tuple, Optional

import cv2
import numpy as np


def resize_image(img: np.ndarray, target_size: Tuple[int, int]):
    """
    Resize image while preserving aspect ratio by adding a black border.
    """
    output_img = np.zeros((target_size[1], target_size[0], img.shape[2]), dtype=np.uint8)
    width_scale = output_img.shape[1] / img.shape[1]
    height_scale = output_img.shape[0] / img.shape[0]
    if width_scale < height_scale:
        target_shape = (int(round(width_scale * img.shape[0])), output_img.shape[1])
    else:
        target_shape = (output_img.shape[0], int(round(height_scale * img.shape[1])))
    x1 = int(round((output_img.shape[1] - target_shape[1]) / 2))
    x2 = x1 + target_shape[1]
    y1 = int(round((output_img.shape[0] - target_shape[0]) / 2))
    y2 = y1 + target_shape[0]
    output_img[y1:y2, x1:x2] = cv2.resize(img, (target_shape[1], target_shape[0]))
    return output_img


class Camera:
    def __init__(self, async_loop: AbstractEventLoop, device_name: str = "/dev/video0",
                 video_buffer_size_s: float = 20.0, resolution: Tuple[int, int] = (640, 480),
                 gain: Optional[float] = None):
        self.__capture = cv2.VideoCapture(device_name)
        self.__capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.__capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        if gain is not None:
            self.__capture.set(cv2.CAP_PROP_GAIN, gain)
        self.__width = int(self.__capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__height = int(self.__capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__framerate = self.__capture.get(cv2.CAP_PROP_FPS)
        frame_buffer_size = int(np.ceil(self.__framerate * video_buffer_size_s))
        self.__frame_buffer = deque(maxlen=frame_buffer_size)
        self.__loop = async_loop
        self.__executor = ThreadPoolExecutor()
        self.__terminate = False
        self.__buffer_lock = Lock()

        self.__read_frames_thread = Thread(target=self.__read_frames)
        self.__read_frames_thread.start()

    def __write_video(self, filename: str, frames: Sequence[np.ndarray],
                      resize_frames: Optional[Tuple[int, int]] = None):
        if resize_frames is not None:
            width, height = resize_frames
        else:
            width = self.__width
            height = self.__height
        writer = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*"mp4v"), self.__framerate, (width, height))
        for f in frames:
            if resize_frames is not None:
                f = resize_image(f, resize_frames)
            writer.write(f)
        writer.release()

    def __read_frames(self):
        while not self.__terminate:
            frame = self.__capture.read()[1]
            with self.__buffer_lock:
                self.__frame_buffer.append(frame)

    async def write_video(self, filename: str, video_sequence_length: Optional[float] = None,
                          resize_frames: Optional[Tuple[int, int]] = None):
        with self.__buffer_lock:
            if video_sequence_length is None:
                frames = list(self.__frame_buffer)
            else:
                frame_count = int(np.ceil(video_sequence_length * self.__framerate))
                frames = [self.__frame_buffer[i] for i in range(max(-frame_count, -len(self.__frame_buffer)), 0)]
        await self.__loop.run_in_executor(self.__executor, self.__write_video, filename, frames, resize_frames)

    def get_current_frame(self) -> np.ndarray:
        if len(self.__frame_buffer) == 0:
            return np.zeros((self.__height, self.__width, 3), dtype=np.uint8)
        return cv2.cvtColor(self.__frame_buffer[-1], cv2.COLOR_BGR2RGB)

    def release(self):
        self.__terminate = True
        self.__capture.release()
        self.__read_frames_thread.join()

    @property
    def video_length_frames(self) -> int:
        return len(self.__frame_buffer)

    @property
    def framerate(self):
        return self.__framerate

    @property
    def auto_exposure(self) -> float:
        return self.__capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)

    @auto_exposure.setter
    def auto_exposure(self, value: float):
        self.__capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)

    @property
    def exposure(self) -> float:
        return self.__capture.get(cv2.CAP_PROP_EXPOSURE)

    @exposure.setter
    def exposure(self, value: float):
        self.__capture.set(cv2.CAP_PROP_EXPOSURE, value)

    @property
    def gain(self) -> float:
        return self.__capture.get(cv2.CAP_PROP_GAIN)

    @gain.setter
    def gain(self, value: float):
        self.__capture.set(cv2.CAP_PROP_GAIN, value)
