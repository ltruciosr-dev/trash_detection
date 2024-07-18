# Trash Detection

Detect recycable items on trash photos.

## Setting Up

0. Create a new environment for yolo.

```bash
conda create -n yolov10 python=3.9
conda activate yolov10
```

1. Download and configure YOLOv10 environment.

```bash
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10
pip install -r requirements.txt
pip install -e .
cd ..
```

2. Download the TACO dataset

```bash
git clone https://github.com/pedropro/TACO
cd TACO
python download.py
cd ..
```

3 . Clone this repository

```bash
git clone https://github.com/ltruciosr-dev/trash_detection.git
cd trash_detection
```

## Format Data

TACO is a dataset structured in COCO format, but in order to train this dataset on YOLO we need to create a `dataset.yaml` file with the next format:

```yaml
path: /home/user/dataset/taco

train: /home/user/dataset/taco/train
val: /home/user/dataset/taco/valid

names: 
  0: category_0
  1: category_1
  2: category_2
  3: category_3
```

In order to do that we have developed a script that will do two things:

1. Select which images will be included on these dataset (based on which categories are needed).
2. Move each acceptable image and create a label file with the fixed bbox (as YOLO requires) and the new categories.

```bash
python scripts/get_data.py
```

## Fine Tune a YOLOv10 model

YOLO is a model developed by the chinesse [Tsinghua University](https://www.tsinghua.edu.cn/en/), the latest development - YOLOv10, is the fastest and more accurated model for object detection and segmentation.

