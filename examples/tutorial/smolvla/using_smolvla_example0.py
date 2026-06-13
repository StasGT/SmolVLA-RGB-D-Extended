import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Backends, Cv2Rotation
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 40


def main():
    device = torch.device("cuda")  # or "cuda" or "cpu" or "mps"
    model_id = "/home/stas/PycharmProjects/vla/lerobot/outputs/train/SmolVLA-RGB-D-Extended/checkpoints/last/pretrained_model"

    model = SmolVLAPolicy.from_pretrained(model_id)

    #print(model)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_id,
        # This overrides allows to run on MPS, otherwise defaults to CUDA (if available)
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # find ports using lerobot-find-port
    follower_port = "/dev/ttyACM0"  # something like "/dev/ttyACT0"

    # the robot ids are used the load the right calibration files
    follower_id = "my_awesome_follower_arm"  # something like "follower_so100"

    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    camera_config = {
        "top": RealSenseCameraConfig(serial_number_or_name="918512072179", fps=15, width=640, height=480,
                                         color_mode=ColorMode.RGB, use_depth=True, rotation=Cv2Rotation.NO_ROTATION)
    }


    robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    robot = SO100Follower(robot_cfg)
    robot.connect()

    task = "Take the matchbox"  # something like "pick the red block"
    robot_type = "so100_follower"  # something like "so100_follower" for multi-embodiment datasets

    # This is used to match the raw observation keys to the keys expected by the policy
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    dataset_features['observation.depth'] = {'dtype': 'depth', 'shape': (480, 640, 3),
                                             'names': ['height', 'width', 'channels']}

    #print(obs_features.keys())

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_features, device=device, task=task, robot_type=robot_type
            )

            obs = preprocess(obs_frame)
            #print(obs.keys())

            action = model.select_action(obs)
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)
            robot.send_action(action)
            #print(obs.keys(), action.keys())

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()
