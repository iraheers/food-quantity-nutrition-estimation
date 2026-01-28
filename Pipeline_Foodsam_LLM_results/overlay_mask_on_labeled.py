from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# -------- GENERAL SETTINGS --------
mask_dir = "/home/sweedal/FoodSAM/results/segmentations/dish_1572021506/sam_mask"
labels_file = "/home/sweedal/FoodSAM/results/segmentations/dish_1572021506/sam_mask_label/semantic_masks_category.txt"
original_img_path = "/home/sweedal/FoodSAM/results/segmentations/dish_1572021506/input.jpg"
output_img_path = "/home/sweedal/FoodSAM/results/segmentations/dish_1572021506/tinted_mask_labels.png"

# Classes you want to tint (if [] = tint all except "background")
TINT_CLASSES = []  # e.g., ["grape", "rice"], or [] for all except "background"
# Set class-color map (add as needed)
CLASS_COLORS = {
    "grape": np.array([0, 255, 0]),     # green
    "rice": np.array([255, 200, 0]),    # yellowish
    "background": np.array([0, 0, 0]),  # will not be shown/tinted
}
DEFAULT_COLOR = np.array([0, 180, 255]) # fallback color (cyan-ish)
ALPHA = 0.35  # Tint strength

# -------- LOAD LABELS --------
labels = []
with open(labels_file) as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split(',')
        labels.append(parts[2])

img = Image.open(original_img_path).convert("RGB")
img_np = np.array(img)

# -------- APPLY TINTED MASKS --------
for i, label in enumerate(labels):
    if label == "background":
        continue
    if TINT_CLASSES and label not in TINT_CLASSES:
        continue
    mask_path = os.path.join(mask_dir, f"{i}.png")
    if not os.path.exists(mask_path):
        continue
    mask = Image.open(mask_path).convert("L").resize(img.size)
    mask_np = np.array(mask)
    tint_color = CLASS_COLORS.get(label, DEFAULT_COLOR)
    for c in range(3):
        img_np[..., c] = np.where(
            mask_np > 0,
            (img_np[..., c] * (1 - ALPHA) + tint_color[c] * ALPHA).astype(np.uint8),
            img_np[..., c]
        )

# -------- OVERLAY LABELS --------
tinted_img = Image.fromarray(img_np)
draw = ImageDraw.Draw(tinted_img)
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

for i, label in enumerate(labels):
    if label == "background":
        continue
    if TINT_CLASSES and label not in TINT_CLASSES:
        continue
    mask_path = os.path.join(mask_dir, f"{i}.png")
    if not os.path.exists(mask_path):
        continue
    mask = Image.open(mask_path).convert("L").resize(img.size)
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 0)
    if len(xs) > 0 and len(ys) > 0:
        centroid_x = int(xs.mean())
        centroid_y = int(ys.mean())
        text = label
        bbox = draw.textbbox((centroid_x, centroid_y), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 180))
        draw.text((centroid_x, centroid_y), text, font=font, fill=(255, 255, 255, 255))

tinted_img.save(output_img_path)
print("Saved tinted mask with class labels:", output_img_path)
