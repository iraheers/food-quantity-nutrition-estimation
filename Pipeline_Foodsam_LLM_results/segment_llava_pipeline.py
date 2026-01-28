import os
import cv2
import requests
import base64
import csv

SEGMENT_ROOT = "/home/sweedal/FoodSAM/results/segmentations/"
OLLAMA_URL = "http://localhost:11434/api/generate"
LAVA_MODEL = "qwen2.5vl:3b"
CSV_SAVE_PATH = "qwen2_5vl_3b_image_results.csv"
PROMPT = (
    "You are an expert nutrition analyst. Analyze the provided food image and estimate its nutritional content. "
    "Reply ONLY with five comma-separated numbers in this order: calories, mass (grams), fat (grams), carbohydrates (grams), protein (grams). "
    "If you are unsure, give your best estimate based on the visible content, and do not write any words, units, or explanation—just five numbers."
)

results = []

for dish_folder in sorted(os.listdir(SEGMENT_ROOT)):
    dish_path = os.path.join(SEGMENT_ROOT, dish_folder)
    if not os.path.isdir(dish_path):
        continue

    # This is the image to send!
    mask_img_path = os.path.join(dish_path, "tinted_mask_labels_resized.jpg")
    if not os.path.exists(mask_img_path):
        print(f"No tinted_mask_labels_resized.jpg found for {dish_folder}")
        continue

    img = cv2.imread(mask_img_path)
    if img is None:
        print(f"Cannot load: {mask_img_path}")
        continue

    # Query LLaVA
    _, buf = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    payload = {
        "model": LAVA_MODEL,
        "prompt": PROMPT,
        "images": [img_b64],
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            caption = resp.json().get("response", "")
        else:
            print(f"LLaVA Error: {resp.status_code} {resp.text}")
            caption = ""
    except Exception as e:
        print("Error with Ollama/LLaVA:", e)
        caption = ""

    results.append([dish_folder, mask_img_path, caption])

# Save results to CSV
with open(CSV_SAVE_PATH, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["dish_folder", "img_path", "llava_caption"])
    writer.writerows(results)

print(f"\nSaved all results to {CSV_SAVE_PATH}")
