# Training Guide

This guide provides detailed information about training models with FreeSeg.

## Table of Contents

- [Training Workflow](#training-workflow)
- [Two-Stage Training](#two-stage-training)
- [Training Options](#training-options)
- [Monitoring Training](#monitoring-training)
- [Resuming Training](#resuming-training)
- [Example Training Commands](#example_training_commands)
- [Next Steps](#next-steps)
- [Troubleshooting](#troubleshooting)

---

## Training Workflow

### Step 1: Prepare Dataset

1. **Organize your data:**
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

2. **Create dataset list file:**
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

**Purpose:** Pre-train the model to predict target values for segmentation labels.

**Configuration:**
```yaml
training:
  wl2_epochs: 5
  wl2_gt_target_value: 15
  pre_train_learning_rate: 0.0001
  wl2_metrics: freeseg.metrics.WeightedL2Loss
```

**What happens:**
- Model learns to predict a target value (e.g., 15) for ground truth labels
- Uses weighted L2 loss
- Typically runs for 5-10 epochs
- Provides better initialization for Dice loss training

### Stage 2: Dice Loss Training

**Purpose:** Fine-tune the model using Dice loss for segmentation.

**Configuration:**
```yaml
training:
  dice_epochs: 100
  learning_rate: 0.0001
  model_metrics: freeseg.metrics.DiceLoss
  dice_squared_form: False
```

**What happens:**
- Model is trained with Dice loss
- Optimizes segmentation accuracy
- Typically runs for 50-200 epochs depending on dataset size

### Training Stages Order

The training automatically proceeds through stages in this order:
1. **wl2** (if `wl2_epochs > 0`)
2. **dice** (if `dice_epochs > 0`)

---

## Training Options

### Basic Options

```bash
--config <config.yaml>
```
Path to configuration file (required).

```bash
--train_output_folder <path>
```
Directory to save training outputs (checkpoints, logs, configs).

```bash
--dataset_list_file <dataset.yaml>
```
Path to dataset list YAML file.

### Model Options

```bash
--model_name <classname>
```
Override model class name from config.

```bash
--crop_size <W H D>
```
Override crop size (must be divisible by `2^(nb_levels)`).

### Training Hyperparameters

```bash
--batch_size <n>
```
Batch size for training.

```bash
--learning_rate <lr>
```
Learning rate for Dice loss training.

```bash
--wl2_epochs <n>
```
Number of weighted L2 pre-training epochs.

```bash
--dice_epochs <n>
```
Number of Dice loss training epochs.

```bash
--steps_per_epoch <n>
```
Number of steps per epoch (default from config).

### Data Loading Options

```bash
--num_workers <n>
```
Number of data loading workers (0 = main process).

```bash
--pin_memory
```
Pin memory for faster GPU transfer.

```bash
--persistent_workers
```
Use persistent workers between epochs.

```bash
--prefetch_factor <n>
```
Prefetch factor for data loading.

```bash
--keep_trainset_in_memory
```
Keep training dataset in memory (requires `--preload`).

### Evaluation Options

```bash
--perform_evaluation
```
Perform evaluation on validation set after each epoch.

```bash
--best_model_metric <loss|dice>
```
Metric to select best model when `--perform_evaluation` is enabled.

### Other Options

```bash
--checkpoint <checkpoint.pt>
```
Resume training from checkpoint.

```bash
--deterministic
```
Use deterministic training (reproducible results).

```bash
--write_tensorboard_summary
```
Write TensorBoard summaries for visualization.

```bash
--cpu
```
Force CPU usage (disable GPU).

```bash
--vmp
```
Enable VMP (Virtual Memory Pool).

```bash
--logfile <logfile>
```
Specify log file path (default: `freeseg_train.log`).

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

### TensorBoard (to be tested)

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
- Learning rate
- Model weights histograms

### Training Output Directory

The training output directory contains:

```
output/training/
  |-- config.<timestamp>.yaml          # Saved configuration
  |-- dataset_list.<timestamp>.yaml     # Saved dataset list
  |-- segmentation_labels.npy          # Segmentation labels
  |-- segmentation_names.npy           # Label names (if provided)
  |-- generation_labels.npy            # Generation labels (if provided)
  |-- topology_classes.npy             # Topology classes (if provided)
  |-- checkpoints/
  |   |-- checkpoint_epoch_<n>.pt       # Epoch checkpoints
  |   |-- best_model.pt                 # Best model (if evaluation enabled)
  |-- tensorboard/                      # TensorBoard logs (if enabled)
  |-- log.<timestamp>                   # Training log
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
  --checkpoint output/training/checkpoints/checkpoint_epoch_50.pt \
  --train_output_folder output/training
```

**What is restored:**
- Model weights
- Optimizer state
- Training epoch
- Best metrics

**Note:** Configuration and dataset list are loaded from the checkpoint, but can be overridden with command-line arguments.

### Checkpoint Information

View checkpoint contents:
```python
from freeseg.checkpoint import Checkpoint
import torch

checkpoint = torch.load("checkpoints/best_model.pt")
Checkpoint.print(checkpoint, detail=True)
```

---
### Debugging Tips

1. **Enable verbose mode:**
   ```bash
   --verbose
   ```

2. **Use CPU for debugging:**
   ```bash
   --cpu
   ```

3. **Check data loading:**
   ```python
   from freeseg.datasets import SegmentationDataset
   # Test dataset loading
   ```

4. **Verify configuration:**
   ```python
   from freeseg.config import Config
   config = Config.load("configs/config.yaml")
   Config.print(config)
   ```

5. **Test augmentation:**
   ```python
   # Test augmentation pipeline
   ```

---

## Example Training Commands

### Basic Training

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --train_output_folder output/training \
  --dataset_list_file data/dataset_list.yaml
```

### Training with Evaluation

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --train_output_folder output/training \
  --dataset_list_file data/dataset_list.yaml \
  --perform_evaluation \
  --best_model_metric dice \
  --write_tensorboard_summary
```

### Training with Custom Hyperparameters

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --train_output_folder output/training \
  --dataset_list_file data/dataset_list.yaml \
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
  --checkpoint output/training/checkpoints/checkpoint_epoch_50.pt \
  --train_output_folder output/training
```

### Training on CPU

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --train_output_folder output/training \
  --dataset_list_file data/dataset_list.yaml \
  --cpu \
  --num_workers 0
```

---

## Next Steps

After training:
1. **Evaluate model:** Use `freeseg_evaluate.py` to compute Dice scores
2. **Predict on new data:** Use `freeseg_predict.py` for inference
3. **Analyze results:** Review TensorBoard logs and evaluation metrics
4. **Fine-tune:** Adjust hyperparameters based on results

See [API.md](API.md) for detailed API documentation and [CONFIGURATION.md](CONFIGURATION.md) for configuration details.

