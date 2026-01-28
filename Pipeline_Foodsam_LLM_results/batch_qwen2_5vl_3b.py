import os
import csv
import torch
from PIL import Image
import re
from transformers import AutoProcessor, AutoModelForVision2Seq

SEGMENT_ROOT = "/home/sweedal/FoodSAM/results/segmentations/"
CSV_SAVE_PATH = "foodqwen2.5vl_3b_results.csv"
MODEL_ID = "AdaptLLM/food-Qwen2.5-VL-3B-Instruct"

PROMPT = (
    "You are a nutrition estimation expert. Carefully analyze the food shown in the image and estimate its nutritional content. "
    "Respond with ONLY five comma-separated numbers, in this exact order: calories, mass in grams, fat in grams, carbohydrates in grams, protein in grams. "
    "IMPORTANT: Do NOT include units, words, or any explanation—ONLY the five numbers, separated by commas. "
    "Do NOT repeat or rephrase the question, do NOT add any extra information, do NOT include your reasoning. "
    "If you cannot estimate a value, write 0 for that value. If the image is unclear, do your best to guess based on what you see. "
    "Your entire reply MUST be ONLY the five comma-separated numbers."
)


def extract_numbers_from_caption(caption):
    # Matches 5 comma-separated numbers (int or float, optional whitespace)
    m = re.search(r'(\d+(\.\d+)?\s*,\s*){4}\d+(\.\d+)?', caption)
    if m:
        return m.group(0).replace(' ', '')
    else:
        return ""  # If no numbers found

# Load model and processor
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="cuda" if torch.cuda.is_available() else "cpu"
)

results = []

for dish_folder in sorted(os.listdir(SEGMENT_ROOT)):
    dish_path = os.path.join(SEGMENT_ROOT, dish_folder)
    if not os.path.isdir(dish_path):
        continue

    img_path = os.path.join(dish_path, "input.jpg")
    if not os.path.exists(img_path):
        print(f"No input.jpg for {dish_folder}")
        continue

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        continue

    # Prepare prompt as chat-style for Qwen
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image"}
        ]}
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
        # Just decode the generated tokens (not input prompt)
        # If output is longer than input_ids, only decode the generated part
        if "input_ids" in inputs and output.shape[1] > inputs["input_ids"].shape[1]:
            gen_tokens = output[0][inputs["input_ids"].shape[1]:]
        else:
            gen_tokens = output[0]
        caption = processor.decode(gen_tokens, skip_special_tokens=True)
        caption_clean = extract_numbers_from_caption(caption)
        print(f"{dish_folder}: {caption_clean}")

    results.append([dish_folder, img_path, caption_clean])

# Save results to CSV
with open(CSV_SAVE_PATH, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["dish_folder", "img_path", "qwen_caption"])
    writer.writerows(results)

print(f"\nSaved all results to {CSV_SAVE_PATH}")

