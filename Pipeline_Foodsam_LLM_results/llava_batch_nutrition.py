import os
import re
import csv
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch

# Path to your segmentations directory
SEGMENTATION_ROOT = '/home/sweedal/FoodSAM/results/segmentations'
OUTPUT_CSV = 'llava_nutrition_predictions.csv'

# Model info
MODEL_ID = "Maressay/llava-v1.6-mistral-7b-hf-bnb-4bit-food-nutrients"
PROMPT = (
    "This is a photo of a meal. "
    "Estimate the calories, mass, fat, carbohydrates, and protein content as accurately as possible. "
    "Reply ONLY with five numbers (decimals allowed), separated by commas, in this exact order: calories, mass, fat, carb, protein. "
    "Do not include any text, units, or explanation. Only the five numbers, separated by commas. "
    "Do not hallucinate."
)

# Load model, tokenizer, processor
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

def extract_macros(text):
    # Use regex to extract numbers for calories, fat, carbs, protein
    # Adjust patterns if the output format differs
    numbers = re.findall(r"(\d+\.?\d*)", text)
    # Try to map in order (Calories, Fat, Carb, Protein)
    result = {"calories": "", "fat": "", "carb": "", "protein": ""}
    if len(numbers) >= 4:
        result["calories"], result["fat"], result["carb"], result["protein"] = numbers[:4]
    return result

results = []
for dish_dir in os.listdir(SEGMENTATION_ROOT):
    dish_path = os.path.join(SEGMENTATION_ROOT, dish_dir)
    image_path = os.path.join(dish_path, "tinted_mask_labels.png")
    if os.path.isdir(dish_path) and os.path.isfile(image_path):
        # Load image
        image = Image.open(image_path).convert("RGB")
        # Prepare input
        inputs = processor(PROMPT, images=image, return_tensors="pt").to(model.device)
        # Generate prediction
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=128)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract macros
        macros = extract_macros(response)
        results.append({
            "dish_id": dish_dir,
            "calories": macros["calories"],
            "fat": macros["fat"],
            "carb": macros["carb"],
            "protein": macros["protein"],
            "llava_output": response
        })
        print(f"Processed {dish_dir} - {macros} - Output: {response}")

# Save all results to CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["dish_id", "calories", "fat", "carb", "protein", "llava_output"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone! Results saved to {OUTPUT_CSV}")
