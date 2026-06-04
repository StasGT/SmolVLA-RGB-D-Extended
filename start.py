from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="my_awesome_follower_arm",
)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyUSB0",
    id="my_awesome_leader_arm",
)

config = OpenCVCameraConfig(
    index_or_path=0,
    fps=15,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)