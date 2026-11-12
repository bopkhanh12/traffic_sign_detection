# Training on TT100K

`annotations_all.json` contains bounding boxes in absolute pixel coordinates.
The training script reads labeled images from `train/`, clips boxes at image
boundaries, creates a reproducible 90/10 train/validation split, and trains
either YOLOv8n or Faster R-CNN (or both). The official `test/` split is not used
for training or validation.

The class map is built from categories that occur in the annotated training
images. This dataset has 173 such classes; categories without training examples
cannot be learned. YOLO-format labels and symlinks to the original images are
created under `prepared/tt100k_yolo/`.

Install dependencies in a virtual environment and train both models for 10
epochs:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
python train_models.py --model both --epochs 10
```

Python 3.12 is required for the PyTorch wheels available on Intel macOS.

Train just one model with `--model yolov8` or `--model faster-rcnn`. Weights
and checkpoints are written under `training_output/`. Pretrained weights are
downloaded on the first run. On a CPU-only machine, training both models may
take a long time. Use `--prepare-only` to prepare and validate the dataset
without importing the training frameworks.

## Local dashboard

Start the local dashboard and image prediction demo:

```bash
python3 app.py
```

Open `http://127.0.0.1:8001`. It reads YOLO metrics from
`training_output/yolov8n/results.csv`; image prediction is enabled after the
first checkpoint is saved.
