import os
import ultralytics
from ultralytics import YOLOv10

# List of models to benchmark
DATA_PATH = "./data/dataset.yaml"
SETTINGS = ultralytics.settings


if __name__ == "__main__":
    # Valid all the trained models.
    runs_dir = f"{SETTINGS['runs_dir']}/detect"

    # List all files in the directory
    directories = os.listdir(runs_dir)

    # Filter files that contain 'train' in their name
    train_dir = [f"{runs_dir}/{dir}" for dir in directories if 'train' in dir]
    
    for dir in train_dir:
        print(f"Validating {dir}")
        model_path = f"{dir}/weights/best.pt"
        if os.path.exists(model_path):
            model = YOLOv10(model_path)
            validation_results = model.val(data=DATA_PATH, imgsz=640, batch=16, conf=0.20, iou=0.6, device="cuda:0")
            print(f"- Validation succesfully generated")
        else:
            print(f"- File not found: {model_path}")