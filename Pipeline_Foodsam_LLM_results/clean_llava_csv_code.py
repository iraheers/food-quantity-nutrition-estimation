import pandas as pd
import re

# Path to your LLaVA CSV output file
input_csv = "/home/sweedal/FoodSAM/foodqwen2.5vl_3b_results.csv"
output_csv = "/home/sweedal/FoodSAM/oodqwen2_5vl_3b_clean.csv"

# Read file
df = pd.read_csv(input_csv)

# Prepare output
rows = []
for idx, row in df.iterrows():
    dish_id = row['dish_folder']
    # Extract only numbers (and decimals) from caption
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(row['qwen_caption']))
    # Pick only the first 5 numbers found (adjust if your format changes)
    if len(nums) >= 5:
        values = nums[:5]
    else:
        # Pad with empty if not enough values
        values = nums + [""] * (5-len(nums))
    rows.append([dish_id] + values)

# Make DataFrame and assign column names
outdf = pd.DataFrame(rows, columns=["dish_id", "calories", "mass", "fat", "carb", "protein"])

# Save to CSV
outdf.to_csv(output_csv, index=False)
print(f"Saved clean CSV to: {output_csv}")
print(outdf.head())
