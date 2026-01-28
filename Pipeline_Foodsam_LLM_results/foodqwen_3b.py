import os
import csv
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Model and paths
model_id = "AdaptLLM/food-Qwen2.5-VL-3B-Instruct"
segment_root = "/home/sweedal/FoodSAM/results/segmentations/"
csv_save_path = "foodqwen_3b_results.csv"
PROMPT = (
    "You are a nutrition analysis AI. Look at the food image and reply with five nutritional values ONLY eparated by commas, DO NOT HALLUCINATE BY GIVING REPEATED VALUES, DO NOT STRICTLY GIVE SENTENCES OR WORDS, in this order: calories, mass in grams, fat in grams, carbohydrates in grams, protein in grams. "
    "Do not write any words, explanations, or sentences. Only output: calories,mass,fat,carbohydrates,protein. "
    "If you don't know, guess based on what you see."
)

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

results = []

for dish_folder in sorted(os.listdir(segment_root)):
    dish_path = os.path.join(segment_root, dish_folder)
    img_path = os.path.join(dish_path, "tinted_mask_labels.png")
    if not os.path.isdir(dish_path) or not os.path.exists(img_path):
        continue

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Could not open {img_path}: {e}")
        continue

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=50)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, ids)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    print(f"{dish_folder}: {output}")
    results.append([dish_folder, img_path, output])

# Save to CSV
with open(csv_save_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["dish_folder", "img_path", "foodqwen3b_output"])
    writer.writerows(results)

print(f"\nSaved all results to {csv_save_path}")

