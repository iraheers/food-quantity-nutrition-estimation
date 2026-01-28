from ultralytics import YOLO

# Load the model
model = YOLO("yolov12x.pt")

# Predict on image
img_path = "C:\\Users\\SWEED\\yolov12\\img1.jpg"
results = model(img_path, task="detect", save=True)

print("Prediction complete!")
print("Output saved in:", results[0].save_dir)

