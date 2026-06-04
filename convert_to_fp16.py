import torch
import os
import json
from safetensors.torch import load_file, save_file

FINAL_CHECKPOINT_DIR = "/home/stas/PycharmProjects/vla/lerobot/outputs/train/SmolVLA-Instruct_2/checkpoints/005000/pretrained_model"
OUTPUT_DIR = "/home/stas/PycharmProjects/vla/lerobot/clean_fp16_smolvla"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Находим и конвертируем сами веса
weights_path = os.path.join(FINAL_CHECKPOINT_DIR, "model.safetensors")
print(f"1. Прямая загрузка бинарных весов из: {weights_path}")
state_dict = load_file(weights_path)

print("2. Низкоуровневая конвертация всех матриц (960 + 720) в FP16 для вашей RTX 2060 Super...")
fp16_state_dict = {}
for key, tensor in state_dict.items():
    # Переводим только дробные веса (веса слоев), оставляя целочисленные индексы (если есть) как было
    if torch.is_floating_point(tensor):
        fp16_state_dict[key] = tensor.to(torch.float16)
    else:
        fp16_state_dict[key] = tensor

output_weights_path = os.path.join(OUTPUT_DIR, "model.safetensors")
print(f"3. Сохранение монолита в {output_weights_path}...")
save_file(fp16_state_dict, output_weights_path)

# 2. Копируем конфигурационные файлы, чтобы они лежали рядом
print("4. Перенос файлов конфигурации...")
config_src = os.path.join(FINAL_CHECKPOINT_DIR, "config.json")
if os.path.exists(config_src):
    with open(config_src, "r") as f:
        cfg_data = json.load(f)
    # Гарантируем, что тип прописан как smolvla для вашего будущего кода
    cfg_data["model_type"] = "smolvla"
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(cfg_data, f, indent=2)

# 3. Сохраняем текстовые и визуальные препроцессоры из облака, обходя баг видеопроцессоров
print("5. Раздельное сохранение токенизатора и процессора картинок...")
from transformers import AutoTokenizer, AutoImageProcessor
base_repo = "HuggingFaceTB/SmolVLM-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_repo)
image_processor = AutoImageProcessor.from_pretrained(base_repo)

tokenizer.save_pretrained(OUTPUT_DIR)
image_processor.save_pretrained(OUTPUT_DIR)

# Записываем простой конфиг-указатель для AutoProcessor
processor_config = {
    "image_processor_type": "Idefics3ImageProcessor",
    "processor_class": "SmolVLMProcessor",
    "tokenizer_class": "LlamaTokenizerFast"
}
with open(os.path.join(OUTPUT_DIR, "processor_config.json"), "w") as f:
    json.dump(processor_config, f, indent=2)

print("✨ УРА! Все барьеры сломаны. Веса переведены в аппаратный FP16 напрямую, минуя кривые абстракции классов!")
