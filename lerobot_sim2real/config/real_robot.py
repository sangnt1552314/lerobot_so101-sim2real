from pathlib import Path
from typing import Optional
import gymnasium as gym
from lerobot.robots.robot import Robot
try:
    from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
except ImportError:
    # lerobot >= 0.4 merged so100/so101 into so_follower
    from lerobot.robots.so_follower import SO100FollowerConfig, SO101FollowerConfig
from lerobot.robots.utils import make_robot_from_config
import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig

def create_real_robot(uid: str = "so100", port: Optional[str] = None, robot_id: Optional[str] = None) -> Robot:
    """Wrapper function to map string UIDS to real robot configurations. Primarily for saving a bit of code for users when they fork the repository. They can just edit the camera, id etc. settings in this one file.

    port / robot_id override the serial port and calibration id configured below (robot_id must match the id used during lerobot calibration)."""
    if uid == "so100":
        robot_config = SO100FollowerConfig(
            port="/dev/ttyACM0",
            id="so100_follower",
            use_degrees=True,
            # for phone camera users you can use the commented out setting below
            # cameras={
            #     "base_camera": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480)
            # }
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            cameras={
                "base_camera": OpenCVCameraConfig(
                    index_or_path=Path("/dev/video2"),
                    height=1080,
                    width=1920,
                    fps=30,
                    warmup_s=2,
                    )
            },
        )
    elif uid == "so101":
        robot_config = SO101FollowerConfig(
            port="/dev/tty.usbmodem5A7A0570661",
            id="so101_follower",
            use_degrees=True,
            # for phone camera users you can use the commented out setting below
            # cameras={
            #     "base_camera": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480)
            # }
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            cameras={
                "base_camera": OpenCVCameraConfig(
                    index_or_path=1,  # iPhone (Continuity Camera); 0 is the MacBook webcam
                    height=1080,
                    width=1920,
                    fps=30,
                    warmup_s=2,
                    )
            },
        )
    else:
        raise ValueError(f"Invalid robot UID: {uid}")
    if port is not None:
        robot_config.port = port
    if robot_id is not None:
        robot_config.id = robot_id
    real_robot = make_robot_from_config(robot_config)
    return real_robot