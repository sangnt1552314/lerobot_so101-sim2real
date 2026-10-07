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
from lerobot.cameras import Cv2Rotation
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
            port="/dev/tty.usbmodem5C821064861",
            id="home_follower",
            use_degrees=True,
            # for phone camera users you can use the commented out setting below
            # cameras={
            #     "base_camera": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480)
            # }
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            cameras={
                "base_camera": OpenCVCameraConfig(
                    # third-view icspring USB camera (640x480). OpenCV indices on macOS shift when cameras are
                    # plugged in/out; at setup time: 0 = wrist icspring (1080p), 1 = this camera, 2 = MacBook FaceTime, 3 = iPhone
                    index_or_path=1,
                    height=480,
                    width=640,
                    fps=30,
                    warmup_s=3,
                    rotation=Cv2Rotation.ROTATE_180,  # the camera is mounted upside down
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
    if uid == "so101":
        # this arm's calibrated shoulder_pan turns opposite to the sim joint (+pan turns the real arm to its
        # left, the sim arm to its right). LeRobotRealAgent negates it when reading and commanding.
        real_robot.sim2real_joint_signs = {"shoulder_pan": -1}
    return real_robot