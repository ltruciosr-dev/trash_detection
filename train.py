import argparse
import subprocess
import ultralytics
from ultralytics import YOLOv10
ultralytics.checks()

# List of models to benchmark
SETTINGS = ultralytics.settings

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Execute YOLO training script with parameters.")
    
    # Add arguments
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--batch', type=int, required=True, help='Batch size')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset YAML file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Construct the command
    command = [
        'yolo',
        f'task=detect',
        f'mode=train',
        f'epochs={args.epochs}',
        f'batch={args.batch}',
        f'plots=True',
        f'model={args.model}',
        f'data={args.data}'
    ]
    
    # Execute the command
    subprocess.run(command)