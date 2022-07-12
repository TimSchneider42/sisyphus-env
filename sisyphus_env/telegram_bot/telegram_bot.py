import asyncio
import json
import os
import time
import traceback
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from tempfile import NamedTemporaryFile
from threading import Thread
from typing import Optional, List, Dict, Tuple

from pyrogram import Client
from pyrogram.raw import functions
from pyrogram.types import Message, InputPhoneContact
from pytgcalls.implementation.group_call import GroupCall

from ..logger import logger
from .telegram_camera_stream import TelegramCameraStream
from .event import Event

API_ID = 8241776
API_HASH = "b2287da4b043b6d943e9dd1af7749643"


async def custom_start_video(group_call: GroupCall, video_stream: TelegramCameraStream):
    if group_call._video_stream and group_call._video_stream.is_running:
        group_call._video_stream.stop()

    group_call._video_stream = video_stream
    group_call._configure_video_capture(group_call._video_stream.get_video_info())

    group_call._video_stream.start()

    group_call._is_video_stopped = False
    if group_call.is_connected:
        await group_call.edit_group_call(video_stopped=False)


def _telegram_bot_process(session_str: str, authorized_users: List[Dict], recv_msg_queue: Queue,
                          cmd_queue: Queue, video_buffer_size_s: float, video_device: str, resolution: Tuple[int, int],
                          resolution_send: Tuple[int, int], auto_exposure: Optional[float], exposure: Optional[float],
                          gain: Optional[float]):
    # Make sure that opencv only gets imported in the telegram process
    import pytgcalls
    from .camera import Camera
    authorized_phone_numbers = {user["phone_number"].strip("+") for user in authorized_users}
    terminate = False

    while not terminate:
        camera = None
        try:
            loop = asyncio.get_event_loop()
            camera = Camera(loop, device_name=video_device, video_buffer_size_s=video_buffer_size_s,
                            resolution=resolution, gain=gain)
            if auto_exposure is not None:
                camera.auto_exposure = auto_exposure
            if exposure is not None:
                camera.exposure = exposure
            write_video_queue = Queue()
            telegram_queue = Queue()

            async def handle_write_video():
                # If this one fails it is safer to re-create the camera instance, hence we let the entire loop fail
                video_terminate = False
                while not video_terminate:
                    while True:
                        try:
                            cmd = write_video_queue.get(block=False)
                            if cmd is None:
                                break
                            else:
                                params = cmd["params"]
                                if cmd["type"] == "store":
                                    await camera.write_video(params["filename"], params["video_length"])
                        except Empty:
                            await asyncio.sleep(0.001)
                    video_terminate = True

            async def handle_telegram():
                tg_terminate = False
                while not tg_terminate:
                    try:
                        video_stream = TelegramCameraStream(camera)

                        async with Client(session_str, API_ID, API_HASH) as client:
                            client: Client
                            group_call = pytgcalls.GroupCallFactory(client).get_group_call()
                            new_contacts = [
                                InputPhoneContact(user["phone_number"], user["first_name"], user.get("last_name", ""))
                                for user in authorized_users]
                            await client.send(functions.contacts.ImportContacts(contacts=new_contacts))

                            @client.on_message()
                            async def handle_message(client: Client, message: Message):
                                if message.from_user.phone_number in authorized_phone_numbers:
                                    if message.service == "voice_chat_started":
                                        await group_call.join(message.chat.id)
                                        await custom_start_video(group_call, video_stream)
                                    else:
                                        recv_msg_queue.put(message)

                            while True:
                                try:
                                    cmd = telegram_queue.get(block=False)
                                    if cmd is None:
                                        break
                                    else:
                                        params = cmd["params"]
                                        if cmd["type"] == "send":
                                            if params["attached_video_sequence_length"] is not None:
                                                while camera.video_length_frames == 0:
                                                    await asyncio.sleep(0.01)
                                                with NamedTemporaryFile(suffix=".mp4") as f:
                                                    await camera.write_video(
                                                        f.name, params["attached_video_sequence_length"],
                                                        resize_frames=resolution_send)
                                                    for uid in authorized_phone_numbers:
                                                        await client.send_video(uid, f.name, caption=params["message"])
                                            else:
                                                for uid in authorized_phone_numbers:
                                                    await client.send_message(uid, params["message"])
                                except Empty:
                                    await asyncio.sleep(0.001)
                        tg_terminate = True
                    except:
                        traceback.print_exc()
                        logger.error("Telegram connection failed. Reestablishing in 5s...")
                        await asyncio.sleep(5.0)

            async def main():
                tg_task = loop.create_task(handle_telegram())
                video_task = loop.create_task(handle_write_video())

                while True:
                    try:
                        cmd = cmd_queue.get(block=False)
                        if cmd is None:
                            write_video_queue.put(None)
                            telegram_queue.put(None)
                            break
                        else:
                            if cmd["type"] == "send":
                                telegram_queue.put(cmd)
                            elif cmd["type"] == "store":
                                write_video_queue.put(cmd)
                    except Empty:
                        await asyncio.sleep(0.001)
                await asyncio.gather(tg_task, video_task)

            loop.run_until_complete(main())
            loop.close()
            terminate = True
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            logger.error("Outer loop caught an exception. Retrying in 5s...")
            time.sleep(5.0)
        finally:
            if camera is not None:
                camera.release()
    logger.info("Telegram bot terminated regularly.")


class TelegramBot:
    def __init__(self, telegram_dir: Path, video_buffer_size_s: float = 20.0, video_device: str = "/dev/video0",
                 video_resolution: Tuple[int, int] = (640, 480),
                 video_resolution_send: Optional[Tuple[int, int]] = None, auto_exposure: Optional[float] = None,
                 exposure: Optional[float] = None, gain: Optional[float] = None):
        with (telegram_dir / "session.txt").open() as f:
            self.__session_str = f.read()
        with (telegram_dir / "authorized_users.json").open() as f:
            self.__authorized_users = json.load(f)
        self.__on_message_event = Event()
        self.__process: Optional[Process] = None
        self.__recv_queue: Optional[Queue] = None
        self.__cmd_queue: Optional[Queue] = None
        self.__video_buffer_size_s = video_buffer_size_s
        self.__video_device = video_device
        self.__thread: Optional[Thread] = None
        self.__video_resolution = video_resolution
        self.__terminate = False
        self.__video_resolution_send = \
            self.__video_resolution if video_resolution_send is None else video_resolution_send
        self.__auto_exposure = auto_exposure
        self.__exposure = exposure
        self.__gain = gain

    def start(self):
        assert self.__process is None, "Already started."
        self.__recv_queue = Queue()
        self.__cmd_queue = Queue()
        self.__terminate = False
        self.__process = Process(
            target=_telegram_bot_process, args=(
                self.__session_str, self.__authorized_users, self.__recv_queue, self.__cmd_queue,
                self.__video_buffer_size_s, self.__video_device, self.__video_resolution, self.__video_resolution_send,
                self.__auto_exposure, self.__exposure, self.__gain),
            daemon=True)
        self.__process.start()
        self.__thread = Thread(target=self.__handle_messages, daemon=True)
        self.__thread.start()

    def stop(self):
        self.__terminate = True
        self.__cmd_queue.put(None)
        self.__thread.join()
        self.__process.join(timeout=30.0)
        if self.__process.exitcode is None:
            logger.info("Killing Telegram process because it did not stop.")
            self.__process.kill()
            self.__process.join()

    def __handle_messages(self):
        while not self.__terminate:
            try:
                message = self.__recv_queue.get(timeout=0.01)
                self.__on_message_event.call(message)
            except Empty:
                pass
            except Exception as ex:
                traceback.print_exc()

    def broadcast_message(self, message: str, attached_video_sequence_length: Optional[float] = None):
        assert attached_video_sequence_length is None or attached_video_sequence_length <= self.__video_buffer_size_s
        cmd = {
            "type": "send",
            "params": {
                "message": message,
                "attached_video_sequence_length": attached_video_sequence_length
            }
        }
        self.__cmd_queue.put(cmd)

    def save_video_sequence(self, filename: os.PathLike, video_length: Optional[float] = None):
        assert video_length is None or video_length <= self.__video_buffer_size_s
        cmd = {
            "type": "store",
            "params": {
                "filename": str(filename),
                "video_length": video_length
            }
        }
        self.__cmd_queue.put(cmd)

    @property
    def on_message_event(self) -> Event:
        return self.__on_message_event

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
