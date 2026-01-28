from groq import Groq
import base64
from PIL import Image

# === Paths ===
input_image_path = r"C:\Users\SWEED\yolov12\runs\detect\predict3\Image.jpg"
resized_image_path = r"C:\Users\SWEED\yolov12\runs\detect\predict3\Image_resized.jpg"

# === Resize/compress image ===
img = Image.open(input_image_path)
img = img.convert("RGB")
img = img.resize((512, 512))
img.save(resized_image_path, "JPEG", quality=85)

# === Encode to base64 ===
def image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

img_base64 = image_to_base64(resized_image_path)
print("Base64 image size:", len(img_base64))

# === API Variables ===
API_KEY = "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"  # Replace with your valid key!
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
PROMPT = "List the food items in this YOLO-detected image and estimate their nutrition."

# === Create Groq client ===
client = Groq(api_key=API_KEY)

# === Send request ===
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
