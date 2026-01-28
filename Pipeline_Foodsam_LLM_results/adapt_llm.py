import os
import csv
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# --- Paths and Model Name ---
SEGMENT_ROOT = "/home/sweedal/FoodSAM/results/segmentations/"
MODEL_NAME = "AdaptLLM/food-Llama-3.2-11B-Vision-Instruct"
CSV_SAVE_PATH = "food_llama_vision_instruct_results.csv"
PROMPT = (
    "Estimate and reply ONLY with four numbers (decimals allowed) separated by commas, representing: calories, protein (g), fat (g), carb (g). "
    "Do not add any units, explanation, or extra text. Only four numbers separated by commas."
)

# --- Load Model and Processor ---
print("Loading model... This may take a while on first run.")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME)

results = []

for dish_folder in sorted(os.listdir(SEGMENT_ROOT)):
    dish_path = os.path.join(SEGMENT_ROOT, dish_folder)
    if not os.path.isdir(dish_path):
        continue

    img_path = os.path.join(dish_path, "tinted_mask_labels_resized.jpg")
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    # Load and process image
    image = Image.open(img_path).convert("RGB")

    # Prepare input
    inputs = processor(images=image, text=PROMPT, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=64)
    result_text = processor.batch_decode(output, skip_special_tokens=True)[0]
    print(f"{dish_folder}: {result_text}")

    results.append([dish_folder, img_path, result_text])

# Save results
with open(CSV_SAVE_PATH, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["dish_folder", "img_path", "nutrition"])
    writer.writerows(results)

print(f"\nSaved all results to {CSV_SAVE_PATH}")
