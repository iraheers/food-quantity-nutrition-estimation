from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

MODEL_ID = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
image_path = "/home/sweedal/FoodSAM/results/segmentations/dish_1558461792/tinted_mask_labels.png"
messages = [
    {"role": "system", "content": "You are a helpful nutrition assistant."},
    {"role": "user", "content": "Given the dish item in the image, reply ONLY with five numbers (decimals allowed) separated by commas, representing: calories,mass,fat,carb,protein. Reply in this exact format: calories,mass,fat,carb,protein. Do not add any text, units, or explanation. Only five numbers separated by commas. Don't hallucinate."}
]

image = Image.open(image_path).convert("RGB")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
if torch.cuda.is_available():
    model = model.to("cuda")

inputs = processor(images=image, messages=messages, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)
    result = processor.batch_decode(output, skip_special_tokens=True)[0].strip()

print("Model output:", result)
