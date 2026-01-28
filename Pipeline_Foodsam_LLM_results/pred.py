import pandas as pd
from sklearn.metrics import mean_absolute_error

# Paths
gt_csv = "/home/sweedal/FoodSAM/nutrition5k_ground_truth_36.csv"
pred_csv = "/home/sweedal/FoodSAM/foodqwen2_5vl_3b_clean.csv"

# Load CSVs
gt = pd.read_csv(gt_csv)
pred = pd.read_csv(pred_csv)

# Convert types (some predictions might be strings)
for col in ["calories", "mass", "fat", "carb", "protein"]:
    pred[col] = pd.to_numeric(pred[col], errors='coerce')
    gt[col] = pd.to_numeric(gt[col], errors='coerce')

# Merge on dish_id
df = pd.merge(gt, pred, on="dish_id", suffixes=('_gt', '_pred'))

# Calculate MAE for each macro
mae_results = {}
for col in ["calories", "mass", "fat", "carb", "protein"]:
    y_true = df[f"{col}_gt"].values
    y_pred = df[f"{col}_pred"].values
    # Only compare non-NaN pairs
    mask = ~pd.isnull(y_true) & ~pd.isnull(y_pred)
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    mae_results[col] = mae

# Print and/or save to file
print("MAE Results:")
for macro, val in mae_results.items():
    print(f"{macro}: {val:.3f}")

# Save as a text file
with open("/home/sweedal/FoodSAM/mae_results.txt", "w") as f:
    for macro, val in mae_results.items():
        f.write(f"{macro}: {val:.3f}\n")

print("MAE results saved to mae_llava_results.txt")

