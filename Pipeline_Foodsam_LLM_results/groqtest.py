from groq import Groq
import base64
from PIL import Image

# === Paths ===
input_image_path = "/home/sweedal/FoodSAM/results/segmentations/dish_1572021506/tinted_mask_labels.png"
resized_image_path = "/home/sweedal/FoodSAM/results/segmentations/dish_1572021506/tinted_mask_labels_resized.jpg"

# === Resize/compress image ===
img = Image.open(input_image_path)
img = img.convert("RGB")
img = img.resize((512, 512))  # You can adjust this if needed
img.save(resized_image_path, "JPEG", quality=85)

# === Encode to base64 ===
def image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

img_base64 = image_to_base64(resized_image_path)
print("Base64 image size:", len(img_base64))

# === API Variables ===
API_KEY = "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"
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
PROMPT = (
    "Given the food item in the image, reply ONLY with four numbers (decimals allowed) separated by commas, representing: calories,mass,fat,carb,protein "
    "Reply in this exact format: calories,mass,fat,carb,protein. "
    "Try to guess the estimates based on the sedgmented masks in the image as shown it can be decimal as well as whole number "
    "Do not add any text, units, or explanation. Only four numbers separated by commas."
)

# === Create Groq client ===
client = Groq(api_key=API_KEY)

# === Send request with correct image_url object format ===
completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                }
            ]
        }
    ],
    temperature=0.3,
    max_tokens=512,
    stream=True,
)

# === Print the streamed response ===
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
