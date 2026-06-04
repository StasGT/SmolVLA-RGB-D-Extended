import os
import json
import torch
import torch.nn as nn
import time
import cv2
import gc
import numpy as np
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoImageProcessor

# Импортируем строго официальные когнитивные и хардверные классы LeRobot
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode

# --- ОФИЦИАЛЬНЫЙ ПАТЧ ДЛЯ ИСПРАВЛЕНИЯ БАГА МАСКИ ВНИМАНИЯ (Long -> Bool) ---
original_eager_attention_forward = SmolVLMWithExpertModel.eager_attention_forward


def patched_eager_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
    if attention_mask is not None: attention_mask = attention_mask.bool()
    return original_eager_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states,
                                            value_states)


SmolVLMWithExpertModel.eager_attention_forward = patched_eager_attention_forward

# --- ОБХОД АССЕРТА PYTORCH ДЛЯ ЭМБЕДДИНГОВ ---
original_embedding_init = nn.Embedding.__init__


def patched_embedding_init(self, num_embeddings, embedding_dim, padding_idx=None, *args, **kwargs):
    if padding_idx is not None and padding_idx >= num_embeddings: padding_idx = None
    return original_embedding_init(self, num_embeddings, embedding_dim, padding_idx, *args, **kwargs)


nn.Embedding.__init__ = patched_embedding_init

# Настройка аллокатора PyTorch под 8 ГБ карту
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POLICY_DTYPE = torch.float16  # Аппаратный FP16 для Tensor Cores карты Turing

CHECKPOINT_DIR = "/home/stas/PycharmProjects/vla/lerobot/outputs/train/SmolVLA-Instruct_2/checkpoints/005000/pretrained_model"

print("1. Создание конфигурации SmolVLA...")
config = SmolVLAConfig()
config.num_steps = 3  # Сжатый шаг денойзинга для скорости
config.use_cache = True
config.device = "cuda"
config.vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
config.load_vlm_weights = False

from lerobot.configs import PolicyFeature, FeatureType

config.input_features = {
    "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
    "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,))
}
config.output_features = {
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))
}

print("2. Инициализация политики SmolVLAPolicy и загрузка весов STRICT=TRUE...")
policy = SmolVLAPolicy(config)
state_dict = load_file(os.path.join(CHECKPOINT_DIR, "model.safetensors"))
fixed_state_dict = {}
for k, v in state_dict.items():
    if not k.startswith("model."):
        fixed_state_dict[f"model.{k}"] = v
    else:
        fixed_state_dict[k] = v
policy.load_state_dict(fixed_state_dict, strict=True)
policy = policy.to(DEVICE, dtype=POLICY_DTYPE)
policy.eval()

print("3. Поднятие лингвистических и визуальных процессоров...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-Instruct", use_fast=True)
image_processor = AutoImageProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

print("4. Загрузка параметров нормализации...")
pre_norm = load_file(os.path.join(CHECKPOINT_DIR, "policy_preprocessor_step_5_normalizer_processor.safetensors"))
post_norm = load_file(os.path.join(CHECKPOINT_DIR, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
state_mean = pre_norm["observation.state.mean"].to(DEVICE, dtype=POLICY_DTYPE)
state_std = pre_norm["observation.state.std"].to(DEVICE, dtype=POLICY_DTYPE)

motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

print("5. Подключение оригинального хардверного драйвера LeRobot...")
try:
    robot_config_path = "/home/stas/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json"
    with open(robot_config_path, "r") as f:
        servo_config = json.load(f)

    # Инициализируем моторы в строгом соответствии со строкой 45 файла so_follower.py
    norm_mode_body = MotorNormMode.RANGE_M100_100
    motors_definition = {
        "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
        "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
        "elbow_flex": Motor(3, "sts3215", norm_mode_body),
        "wrist_flex": Motor(4, "sts3215", norm_mode_body),
        "wrist_roll": Motor(5, "sts3215", norm_mode_body),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    loaded_calibration = {}
    for m_name, m_cal in servo_config.items():
        loaded_calibration[m_name] = MotorCalibration(
            id=m_cal["id"], drive_mode=m_cal["drive_mode"], homing_offset=m_cal["homing_offset"],
            range_min=m_cal["range_min"], range_max=m_cal["range_max"]
        )
    motors_bus = FeetechMotorsBus(port="/dev/ttyACM0", motors=motors_definition, calibration=loaded_calibration)
    motors_bus.connect()
    motors_bus.write_calibration(loaded_calibration, cache=True)
    with motors_bus.torque_disabled():
        motors_bus.configure_motors()
        for m_name in motors_bus.motors:
            motors_bus.write("Operating_Mode", m_name, 0)
            motors_bus.write("P_Coefficient", m_name, 16)
    motors_bus.enable_torque()
    print("✅ Реальный драйвер LeRobot полностью готов!")
except Exception as e:
    print(f"⚠️ Ошибка инициализации драйвера: {e}. Симуляция.")
    motors_bus = None


# --- ИСТИННЫЙ АЛГОРИТМ ПРЕПРОЦЕССИНГА АВТОРОВ (LETTERBOX НА МАСКУ ИЗ НУЛЕЙ 512x512) ---
def author_letterbox_preprocessor(frame, target_size=512):
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    mask[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return mask


# --- АСИНХРОННЫЕ КАМЕРЫ ---
class AsyncWristCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.latest_frame = None
        self.is_running = True
        import threading
        threading.Thread(target=self._loop, daemon=True).start()
        while self.latest_frame is None: time.sleep(0.1)

    def _loop(self):
        while self.is_running:
            if self.cap.grab():
                ret, f = self.cap.retrieve()
                if ret: self.latest_frame = f
            time.sleep(0.01)

    def get_frame(self):
        return self.latest_frame

    def stop(self):
        self.is_running = False
        if self.cap: self.cap.release()


class AsyncTopRealsenseCamera:
    def __init__(self, serial_number="918512072179"):
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.latest_frame = None
        self.is_running = True
        import threading
        threading.Thread(target=self._loop, daemon=True).start()
        while self.latest_frame is None: time.sleep(0.1)

    def _loop(self):
        while self.is_running:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame: self.latest_frame = np.asanyarray(color_frame.get_data())
            time.sleep(0.01)

    def get_frame(self):
        return self.latest_frame

    def stop(self):
        self.is_running = False
        self.pipeline.stop()


print("5. Инициализация видеопотоков...")
cam_wrist = AsyncWristCamera()
try:
    cam_top = AsyncTopRealsenseCamera("918512072179")
    print("✅ Обе камеры успешно активированы!")
except Exception as e:
    print(f"⚠️ Ошибка RealSense: {e}. Дублируем wrist.")
    cam_top = cam_wrist

# Инициализируем обратную связь в каноничном интервале [-1.0, 1.0]
current_robot_angles = np.zeros(6, dtype=np.float32)
action_buffer = None

print("\n🤖 ПОЕХАЛИ! Прямой инференс с честной нормализацией по коду авторов...")
try:
    while True:
        t_start = time.perf_counter()

        raw_top = cam_top.get_frame()
        raw_wrist = cam_wrist.get_frame()

        frame_top_512 = author_letterbox_preprocessor(raw_top)
        frame_wrist_512 = author_letterbox_preprocessor(raw_wrist)

        # ОПРОС ЖЕЛЕЗА ПО КОДУ АВТОРОВ (Строка 166 so_follower.py):
        # Драйвер возвращает текущие углы СРАЗУ в интервале [-1.0, 1.0]
        if motors_bus is not None:
            present_pos_dict = motors_bus.sync_read("Present_Position")
            current_robot_angles = np.array([present_pos_dict[name] for name in motor_names], dtype=np.float32)

        raw_state = torch.from_numpy(current_robot_angles).to(DEVICE, dtype=POLICY_DTYPE)
        norm_state = (raw_state - state_mean) / (state_std + 1e-7)

        padded_state = torch.zeros(32, dtype=POLICY_DTYPE, device=DEVICE)
        padded_state[:6] = norm_state
        padded_state = padded_state.unsqueeze(0)

        # ОФИЦИАЛЬНАЯ ПРЕПОДГОТОВКА ИЗОБРАЖЕНИЙ ПО КОДУ LEROBOT (RGB + CHW / 255.0)
        rgb_top = cv2.cvtColor(frame_top_512, cv2.COLOR_BGR2RGB)
        rgb_wrist = cv2.cvtColor(frame_wrist_512, cv2.COLOR_BGR2RGB)

        tensor_top = torch.from_numpy(rgb_top).permute(2, 0, 1).float() / 255.0
        tensor_wrist = torch.from_numpy(rgb_wrist).permute(2, 0, 1).float() / 255.0

        img_mean = torch.tensor([0.48145103, 0.4578275, 0.40821073]).view(3, 1, 1)
        img_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

        tensor_top = (tensor_top - img_mean) / img_std
        tensor_wrist = (tensor_wrist - img_mean) / img_std

        img_top_ready = tensor_top.unsqueeze(0).to(DEVICE, dtype=POLICY_DTYPE)
        img_wrist_ready = tensor_wrist.unsqueeze(0).to(DEVICE, dtype=POLICY_DTYPE)

        text_inputs = tokenizer("Take the matchbox", return_tensors="pt")

        batch = {
            "observation.state": padded_state,
            "observation.images.top": img_top_ready,
            "observation.images.wrist": img_wrist_ready,
            "observation.language.tokens": text_inputs["input_ids"].to(DEVICE),
            "observation.language.attention_mask": text_inputs["attention_mask"].to(DEVICE)
        }

        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda", dtype=POLICY_DTYPE):
                outputs = policy.predict_action_chunk(batch)
                predicted_chunk = outputs.squeeze(0).to(torch.float32).cpu().numpy()

        if action_buffer is None:
            action_buffer = predicted_chunk[:16, :]
        else:
            k = 0.35
            action_buffer = (1 - k) * action_buffer + k * predicted_chunk[:16, :]

        # Извлекаем чистые нормализованные выходы ИИ [-1.0, 1.0]
        next_step_actions = action_buffer[1, :]

        # --- СУЩЕСТВУЮЩИЙ АЛГОРИТМ ОТПРАВКИ АВТОРОВ (Строка 183 so_follower.py) ---
        if motors_bus is not None:
            # Передаем чистые значения [-1, 1] напрямую в словарь БЕЗ умножений на 100!
            goal_pos = {name: float(next_step_actions[i]) for i, name in enumerate(motor_names)}

            # Драйвер сам переведет этот интервал в тики, не утыкаясь в лимиты безопасности
            motors_bus.sync_write("Goal_Position", goal_pos)
            print(f"🎯 [OFFICIAL PIPELINE] Отправлен goal_pos: {np.round(next_step_actions, 2)}")

        action_buffer = np.roll(action_buffer, -1, axis=0)
        action_buffer[-1, :] = predicted_chunk[15, :]

        duration = time.perf_counter() - t_start
        print(f"🎯 Выполнено! Скорость: {1.0 / duration:.1f} FPS | Задержка: {duration * 1000:.1f} мс")
        torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\n🛑 Инференс остановлен.")
finally:
    cam_wrist.stop()
    if 'cam_top' in locals() and hasattr(cam_top, 'stop'): cam_top.stop()
    if motors_bus is not None: motors_bus.disconnect()
