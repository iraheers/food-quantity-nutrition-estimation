import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

image_path = "C:/Users/SWEED/yolov12/runs/detect/predict3/Image.jpg"
image = Image.open(image_path)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "List all items in the image and their protein, carbohydrate, fat per 1000 grams and overall calories."},
            {"type": "image"}
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)

outputs = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(outputs[0], skip_special_tokens=True))
