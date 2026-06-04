
"""
hf auth login --token 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 --add-to-git-credential

export HF_TOKEN="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"

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{
    wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, warmup_s: 5},
    top: {type: intelrealsense, serial_number_or_name: 918512072179, width: 640, height: 480, fps: 30, use_depth: True}
    }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.root==/home/stas/.cache/huggingface/lerobot/StasGT/record-test1_20260602_133845 \
    --dataset.repo_id=StasGT/record-test \
    --dataset.push_to_hub=true \
    --dataset.private=true \
    --dataset.num_episodes=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Take a matchbox and place it on the box"

hf upload StasGT/record-test ~/.cache/huggingface/lerobot/StasGT/record-test1_20260602_140917 --repo-type dataset

lerobot-train \
  --policy.type=smolvla \
  --dataset.repo_id=StasGT/record-test \
  --dataset.root=/home/stas/.cache/huggingface/lerobot/StasGT/record-test1_20260602_140917 \
  --output_dir=~/lerobot/outputs/train/SmolVLA-RGB-D-Extended \
  --job_name=SmolVLA-RGB-D-Extended \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=1 \
  --policy.repo_id=StasGT/SmolVLA-RGB-D-Extended
  --policy.push_to_hub=false


lerobot-train \
 --policy.type=smolvla \
 --policy.pretrained_path=lerobot/smolvla_base \
 --dataset.repo_id=StasGT/record-test \
 --dataset.root=/home/stas/.cache/huggingface/lerobot/StasGT/record-test_20260515_171305 \
 --policy.repo_id=StasGT/SmolVLA-Instruct_1 \
 --output_dir=outputs/train/SmolVLA-Instruct_1 \
 --job_name=SmolVLA-Instruct_1 \
 --policy.device=cuda \
 --policy.optimizer_lr=1e-3 \
 --policy.scheduler_decay_lr=1e-4 \
 --steps=5000 \
 --save_checkpoint=true \
 --save_freq=500 \
 --batch_size=24 \
 --peft.method_type=LORA \
 --peft.r=16



lerobot-train \
 --resume=true \
 --policy.dtype=float16 \
 --config_path=outputs/train/SmolVLA-Instruct_2/checkpoints/005000/pretrained_model/train_config.json \
 --policy.type=smolvla \
 --policy.pretrained_path=lerobot/smolvla_base \
 --dataset.repo_id="StasGT/record-test" \
 --dataset.root="/home/stas/.cache/huggingface/lerobot/StasGT/record-test_20260515_171305" \
 --output_dir="outputs/train/SmolVLA-Instruct_2" \
 --policy.repo_id=StasGT/SmolVLA-Instruct_2 \
 --policy.device=cuda \
 --steps=5500 \
 --save_checkpoint=true \
 --save_freq=500 \
 --batch_size=24 \
 --policy.optimizer_lr=1e-3 \
 --policy.scheduler_decay_lr=1e-4



  --resume=true

  --policy.path=outputs/train/smolvla_so101_test/checkpoints/last/pretrained_model

  --inference.type=rtc \
  --inference.rtc.execution_horizon=20 \
  --inference.rtc.max_guidance_weight=5.0 \
  --inference.rtc.prefix_attention_schedule=LINEAR \


lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/SmolVLA-Instruct_2/checkpoints/005000/pretrained_model \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{
  wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, warmup_s: 5},
  top: {type: intelrealsense, serial_number_or_name: 918512072179, width: 640, height: 480, fps: 30, use_depth: False}}" \
  --task="Take a matchbox" \
  --duration=60

"""

import  cv2
cv2.findChessboardCorners

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Construct an `OpenCVCameraConfig` with your desired FPS, resolution, color mode, and rotation.
config = OpenCVCameraConfig(
    index_or_path=0,
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION,
    warmup_s = 3
)

# Instantiate and connect an `OpenCVCamera`, performing a warm-up read (default).
with OpenCVCamera(config) as camera:

    # Read a frame synchronously — blocks until hardware delivers a new frame
    frame = camera.read()
    print(f"read() call returned frame with shape:", frame.shape)

    # Read a frame asynchronously with a timeout — returns the latest unconsumed frame or waits up to timeout_ms for a new one
    try:
        for i in range(10):
            frame = camera.async_read(timeout_ms=200)
            print(f"async_read call returned frame {i} with shape:", frame.shape)
    except TimeoutError as e:
        print(f"No frame received within timeout: {e}")

    # Instantly return a frame - returns the most recent frame captured by the camera
    try:
        initial_frame = camera.read_latest(max_age_ms=1000)
        for i in range(10):
            frame = camera.read_latest(max_age_ms=1000)
            print(f"read_latest call returned frame {i} with shape:", frame.shape)
            print(f"Was a new frame received by the camera? {not (initial_frame == frame).any()}")
    except TimeoutError as e:
        print(f"Frame too old: {e}")