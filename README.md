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

Make sure that a `data` directory is created and inside we have 2 directories, `train` and `val`, and 1 file, the `dataset.yaml` file.

## Fine-tune a YOLOv10 model

YOLO is a model developed by the chinesse [Tsinghua University](https://www.tsinghua.edu.cn/en/), the latest development - YOLOv10, is the fastest and more accurated model for object detection and segmentation.

1. Download the YOLOv10 weights

Here select which weights to download.

```bash
python scripts/get_data.py
```

2. Tune the selected model

```bash
python train.py --epochs 5 --batch 16 --model weights/yolov10s.pt --data data/dataset.yaml
```

## Validate the tuned model

1. Check the training runs

All the models area store on the `runs_dir` directory settled when installing YOLO, to check the directory check on:

```bash
python -c "import ultralytics; from pprint import pprint; pprint(ultralytics.settings)"
```

The output will be:

```yaml
{
  'api_key': '',
  'clearml': True,
  'comet': True,
  'datasets_dir': '/home/ltruciosr/git/datasets',
  'dvc': True,
  'hub': True,
  'mlflow': True,
  'neptune': True,
  'openai_api_key': '',
  'raytune': True,
  'runs_dir': '/home/ltruciosr/git/yolov10/runs',
  'settings_version': '0.0.4',
  'sync': True,
  'tensorboard': True,
  'uuid': '5e42f2651d8fe9fce5d35c16ff21f88830a7b94c0efc01e29e3ad3162582d562',
  'wandb': True,
  'weights_dir': '/home/ltruciosr/git/yolov10/weights'
}
```

2. Perform evaluations

The next script will iterate over all the runs and perform validations for each model.

```bash
python scripts/valid.py
```