from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# --- Paths ---
original_img_path = "/home/sweedal/FoodSAM/results/segmentations/dish_1558549150/input.jpg"
mask_folder = "/home/sweedal/FoodSAM/results/segmentations/dish_1558549150/sam_mask"
labels_file = "/home/sweedal/FoodSAM/results/segmentations/dish_1558549150/sam_mask_label/semantic_masks_category.txt"
output_img_path = "/home/sweedal/FoodSAM/results/segmentations/dish_1558549150/labeled_full_image.png"

# --- Load labels ---
labels = []
with open(labels_file, 'r') as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split(',')
        labels.append(parts[2])

# --- Load original image ---
orig_img = Image.open(original_img_path).convert("RGB")
out_img = orig_img.copy()

try:
    font = ImageFont.truetype("arial.ttf", 32)
except:
    font = ImageFont.load_default()

# --- Overlay labels on each region ---
for i, label in enumerate(labels):
    mask_path = os.path.join(mask_folder, f"{i}.png")
    if not os.path.exists(mask_path):
        continue
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    coords = np.argwhere(mask_np > 0)
    if coords.size == 0:
        continue
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Place label at the top-left corner of the region (x0, y0)
    draw = ImageDraw.Draw(out_img)
    text = label

    # Get text size
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = font.getsize(text)

    # Draw black rectangle background for text
    rect_x0, rect_y0 = x0, y0
    rect_x1, rect_y1 = x0 + text_width + 10, y0 + text_height + 10
    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(0, 0, 0, 180))

    # Draw text on top
    draw.text((x0 + 5, y0 + 5), text, font=font, fill=(255, 255, 255, 255))

# --- Save final labeled image ---
out_img.save(output_img_path)
print(f"Labeled image saved to: {output_img_path}")
