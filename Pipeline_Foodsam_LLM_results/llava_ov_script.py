import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
import os
import csv
import re

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
segment_root = "/home/sweedal/FoodSAM/results/segmentations/"
csv_save_path = "llava_ov_mask_results.csv"

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

prompt = (
    "You are an experienced nutrition analyst. Carefully analyze the food image and estimate its nutritional content. "
    "Respond ONLY with five decimal numbers, separated by commas, in this exact order: calories, mass, fat, carbs, protein."
)

results = []

for dish_folder in sorted(os.listdir(segment_root)):
    dish_path = os.path.join(segment_root, dish_folder)
    if not os.path.isdir(dish_path):
        continue

    img_path = os.path.join(dish_path, "tinted_mask_labels.png")
    if not os.path.exists(img_path):
        continue

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Could not open {img_path}: {e}")
        continue

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"}
            ],
        },
    ]
    chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=chat_prompt, return_tensors="pt").to(0, torch.float16)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
        # Use decode without slicing tokens!
        decoded = processor.decode(output[0], skip_special_tokens=True).strip()
        # Extract only the first 5 numbers (as string, comma-separated)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", decoded)
        if len(numbers) == 5:
            formatted = ",".join(numbers)
        else:
            formatted = decoded  # For debugging/inspection

    results.append([dish_folder, img_path, formatted])
    print(f"{dish_folder}: {formatted}")

with open(csv_save_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["dish_folder", "img_path", "llava_output"])
    writer.writerows(results)

print(f"\nSaved all results to {csv_save_path}")
