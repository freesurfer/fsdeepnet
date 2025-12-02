# Training Guide

This guide provides detailed information about training models with FreeSeg.

## Table of Contents

- [Training Workflow](#training-workflow)
- [Two-Stage Training](#two-stage-training)
- [Monitoring Training](#monitoring-training)
- [Resuming Training](#resuming-training)
- [Example Training Commands](#example-training-commands)

---

## Training Workflow

### Step 1: Prepare Dataset

- **Organize your data:**
   ```
   data_folder/
     |-- images/
     |   |-- subject001.mgz
     |   |-- subject002.mgz
     |   |-- ...
     |-- labels/
     |   |-- subject001.mgz
     |   |-- subject002.mgz
     |   |-- ...
     |-- priors/  # optional
     |   |-- subject001.mgz
     |   |-- ...
   ```

- **Create dataset list file:**
   ```yaml
   train:
     - image_filepath: /path/to/images/subject001.mgz
       label_filepath: /path/to/labels/subject001.mgz
   validation:
     - image_filepath: /path/to/images/subject101.mgz
       label_filepath: /path/to/labels/subject101.mgz
   test:
     - image_filepath: /path/to/images/subject201.mgz
       label_filepath: /path/to/labels/subject201.mgz
   ```

### Step 2: Configure Training

Create or edit `config.yaml` (see [CONFIGURATION.md](CONFIGURATION.md) for details).

### Step 3: Run Training

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --train_output_folder output/training \
  --dataset_list_file data/dataset_list.yaml
```

---

## Two-Stage Training

FreeSeg supports a two-stage training approach for better model initialization.

### Stage 1: Weighted L2 Pre-training

**Purpose:** Pre-train the model with weighted L2 norm loss function to provide better initialization for Dice loss training

**Parameters:**
```yaml
training:
  wl2_epochs: 5
  wl2_gt_target_value: 15
  pre_train_learning_rate: 0.0001
  wl2_metrics: freeseg.metrics.WeightedL2Loss
```

### Stage 2: Dice Loss Training

**Purpose:** Fine-tune the model using soft Dice loss function

**Parameters:**
```yaml
training:
  dice_epochs: 100
  learning_rate: 0.0001
  model_metrics: freeseg.metrics.DiceLoss
  dice_squared_form: False
```

### Training Stages Order

The training automatically proceeds through stages in this order:
1. **wl2** (if `wl2_epochs > 0`)
2. **dice** (if `dice_epochs > 0`)

---

## Monitoring Training

### Log Files

Training logs are saved to:
- Default: `freeseg_train.log` in current directory
- Or: Path specified by `--logfile`

**Log contents:**
- Training command and configuration
- Training progress (loss, dice scores)
- Validation metrics (if evaluation enabled)
- Model checkpoint information

### TensorBoard  (??? to be tested)

Enable TensorBoard logging:
```bash
--write_tensorboard_summary
```

View training metrics:
```bash
tensorboard --logdir output/training
```

**Available metrics:**
- Training loss
- Training Dice scores
- Validation loss (if evaluation enabled)
- Validation Dice scores (if evaluation enabled)

### Training Output Directory

The training output directory contains:

```
output/training/
  |-- config.<timestamp>.yaml          # Saved configuration
  |-- dataset_list.<timestamp>.yaml    # Saved dataset list
  |-- segmentation_labels.npy          # Segmentation labels
  |-- segmentation_names.npy           # Label names (if provided)
  |-- generation_labels.npy            # Generation labels (if provided)
  |-- topology_classes.npy             # Topology classes (if provided)
  |-- wl2_<n>.pth                      # wl2 pre-training checkpoints (if wl2 enabled)
  |-- dice_<n>.pth                     # dice training checkpoints
  |-- best_model.pth                   # (??? double check) Best model (if evaluation enabled)
  |-- tensorboard/                     # (??? double check) TensorBoard logs (if enabled)
  |-- log.<timestamp>                  # Training log
```

### Checkpoint Files

Checkpoints contain:
- Model state dictionary
- Optimizer state dictionary
- Current epoch
- Best loss and Dice scores
- Model architecture dictionary
- Dataset configuration
- Label lookup table

---

## Resuming Training

### From Checkpoint

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --dataset_list_file data/dataset_list.yaml \  
  --train_output_folder output/training \
  --checkpoint output/training/checkpoints/checkpoint_epoch_50.pth  
```

### Checkpoint Information

View checkpoint contents:
```python
from freeseg.checkpoint import Checkpoint
import torch

checkpoint = torch.load("checkpoints/best_model.pth")
Checkpoint.print(checkpoint, detail=True)
```
---

## Example Training Commands

### Basic Training

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --dataset_list_file data/dataset_list.yaml \
  --train_output_folder output/training \
```

### Training with Evaluation

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --dataset_list_file data/dataset_list.yaml \  
  --train_output_folder output/training \
  --perform_evaluation \
  --best_model_metric dice \
  --write_tensorboard_summary
```

### Training with Custom Hyperparameters

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --dataset_list_file data/dataset_list.yaml \  
  --train_output_folder output/training \
  --batch_size 2 \
  --learning_rate 0.0005 \
  --dice_epochs 150 \
  --wl2_epochs 10 \
  --crop_size 128 128 128
```

### Resuming Training

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --dataset_list_file data/dataset_list.yaml \
  --train_output_folder output/training \
  --checkpoint output/training/checkpoints/checkpoint_epoch_50.pth  
```

### Training on CPU

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --dataset_list_file data/dataset_list.yaml \  
  --train_output_folder output/training \
  --cpu
```

---

See [API.md](API.md) for detailed API documentation and [CONFIGURATION.md](CONFIGURATION.md) for configuration details.

