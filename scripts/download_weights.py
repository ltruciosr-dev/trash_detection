import os
import subprocess
import ultralytics
ultralytics.checks()

# Create the directory
subprocess.run(['mkdir', '-p', 'weights'])

# List of URLs to download
urls = [
    'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt',
    'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt',
    'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt',
    'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt',
    'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt',
    'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt'
]

if __name__ == '__main__':
    # Get the list of existing files in the weights directory
    existing_files = os.listdir('weights')

    # Download the files if they don't already exist
    for url in urls:
        filename = url.split('/')[-1]
        if filename not in existing_files:
            print(f"Downloading {filename}...")
            subprocess.run(['wget', '-P', 'weights', '-q', url])
        else:
            print(f"{filename} already exists, skipping download.")

    # List the contents of the weights directory
    subprocess.run(['ls', '-lh', 'weights'])