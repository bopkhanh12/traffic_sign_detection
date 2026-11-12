# TT100K on Kaggle

This archive contains `train_models.py`, `annotations_all.json`, the labeled
training images, and the pretrained YOLOv8n starting weights. Test and
unlabeled images are omitted.

## Run with a Kaggle GPU

1. Create a Kaggle Dataset from `tt100k_kaggle.zip` and attach it to a notebook.
2. In the notebook settings, select **GPU** as the accelerator.
3. Extract and train YOLOv8n (Kaggle's image already includes PyTorch):

```python
!unzip -q /kaggle/input/tt100k-kaggle/tt100k_kaggle.zip -d /kaggle/working
%cd /kaggle/working/tt100k_kaggle
import torch
print("CUDA available:", torch.cuda.is_available())
!python train_models.py --model yolov8 --epochs 10
```

If the dataset mount name differs, replace `/kaggle/input/tt100k-kaggle` with
the folder shown under `/kaggle/input`.

The training script automatically selects CUDA, then Apple MPS, then CPU. It
creates the YOLO labels and train/validation split on first run. Results are
written to `training_output/`; save that folder as a Kaggle notebook output to
keep the checkpoints after the session ends.
