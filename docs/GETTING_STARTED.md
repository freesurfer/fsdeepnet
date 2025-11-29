# Getting Started with FreeSeg

This guide will help you get started with FreeSeg for medical image segmentation.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Your First Training](#your-first-training)
- [Your First Prediction](#your-first-prediction)
- [Next Steps](#next-steps)

---

## Installation

### Prerequisites

1. **FreeSurfer**: FreeSeg requires FreeSurfer to be installed and configured
2. **Python**: Python 3.8 (via FreeSurfer's fspython)
3. **CUDA**: GPU with CUDA support (optional but recommended)

### Installation Steps

1. **Build FreeSurfer with Python support:**
   ```bash
   # Build FreeSurfer with cmake options:
   cmake -DDISTRIBUTE_FSPYTHON=ON -DFSPYTHON_INSTALL_CUDA=ON ...
   ```

2. **Source FreeSurfer environment:**
   ```bash
   source $FREESURFER_HOME/SetUpFreeSurfer.sh
   ```

3. **Clone the repository:**
   ```bash
   git clone https://github.com/freesurfer/freeseg.git
   cd freeseg
   ```

4. **Install FreeSeg:**
   ```bash
   # Standard installation
   fspython -m pip install .
   
   # Editable installation (for development)
   fspython -m pip install --editable .
   ```

### Verify Installation

```bash
# Check if freeseg is installed
fspython -c "import freeseg; print(freeseg.__file__)"

# Check if scripts are available
which freeseg_train.py
which freeseg_predict.py
which freeseg_evaluate.py
```

---

## Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:

```
my_data/
  |-- images/
  |   |-- subject001.mgz
  |   |-- subject002.mgz
  |   |-- ...
  |-- labels/
  |   |-- subject001.mgz
  |   |-- subject002.mgz
  |   |-- ...
```

### 2. Create Dataset List

Create a YAML file (`dataset_list.yaml`):

```yaml
train:
  - image_filepath: /path/to/my_data/images/subject001.mgz
    label_filepath: /path/to/my_data/labels/subject001.mgz
  - image_filepath: /path/to/my_data/images/subject002.mgz
    label_filepath: /path/to/my_data/labels/subject002.mgz

validation:
  - image_filepath: /path/to/my_data/images/subject101.mgz
    label_filepath: /path/to/my_data/labels/subject101.mgz

test:
  - image_filepath: /path/to/my_data/images/subject201.mgz
    label_filepath: /path/to/my_data/labels/subject201.mgz
```

### 3. Create Configuration File

Copy the example configuration:

```bash
cp configs/config.yaml my_config.yaml
```

Edit `my_config.yaml`:

```yaml
dataset:
  dataset_classname: freeseg.datasets.segmentationdataset.SegmentationDataset
  segmentation_labels: [0, 2, 3, 4, 17, 41]  # Your label values
  expected_num_channels: 1
  dataset_list_file: /path/to/dataset_list.yaml

model:
  name: freeseg.models.unet.UNet
  nb_levels: 3
  nb_features: 24
  ndims: 3
  crop_size: [160, 160, 160]

preprocessing:
  crop_size: [160, 160, 160]
  augmentations:
    - spatialdeformation:
        affine_probability: 1.0
        max_translation: 30
        max_rotation: 20
    - centercrop:
        max_offset: [1, 2, 3]
    - intensityaugmentation:
        normalize: True

training:
  batch_size: 1
  dice_epochs: 50
  learning_rate: 0.0001
  steps_per_epoch: 100
```

### 4. Train Your Model

```bash
fspython scripts/freeseg_train.py \
  --config my_config.yaml \
  --train_output_folder output/my_training \
  --dataset_list_file /path/to/dataset_list.yaml
```

### 5. Predict on New Images

```bash
fspython scripts/freeseg_predict.py \
  --i /path/to/new_image.mgz \
  --o /path/to/output_segmentation.mgz \
  --checkpoint output/my_training/checkpoints/best_model.pt
```

---

## Your First Training

### Step-by-Step Training

#### Step 1: Prepare Dataset

1. **Organize data:**
   - Place images in `data/images/`
   - Place labels in `data/labels/`
   - Ensure matching filenames

2. **Create dataset list:**
   ```bash
   # Use the helper script (if available)
   fspython scripts/freeseg_create_data_list.py \
     -d data/ \
     -o dataset_list.yaml
   ```

   Or create manually (see [Quick Start](#quick-start))

#### Step 2: Configure Training

1. **Copy example config:**
   ```bash
   cp configs/config.yaml my_config.yaml
   ```

2. **Edit configuration:**
   - Set `segmentation_labels` to your label values
   - Set `dataset_list_file` to your dataset list path
   - Adjust `crop_size` based on your image size and GPU memory
   - Adjust `batch_size` (start with 1 for 3D images)
   - Set `dice_epochs` (start with 50-100)

#### Step 3: Start Training

```bash
fspython scripts/freeseg_train.py \
  --config my_config.yaml \
  --train_output_folder output/first_training \
  --dataset_list_file dataset_list.yaml \
  --write_tensorboard_summary
```

#### Step 4: Monitor Training

1. **Check log file:**
   ```bash
   tail -f output/first_training/log.*
   ```

2. **View TensorBoard:** (to be tested)
   ```bash
   tensorboard --logdir output/first_training
   ```
   Open browser to `http://localhost:6006`

3. **Check checkpoints:**
   ```bash
   ls output/first_training/checkpoints/
   ```

#### Step 5: Evaluate Model

```bash
fspython scripts/freeseg_evaluate.py \
  --gt /path/to/ground_truth_folder \
  --seg /path/to/predicted_segmentations_folder \
  --segmentation_labels output/first_training/segmentation_labels.npy
```

---

## Your First Prediction

### Step 1: Load Trained Model

Make sure you have a trained checkpoint:
```bash
ls output/first_training/checkpoints/best_model.pt
```

### Step 2: Run Prediction

```bash
fspython scripts/freeseg_predict.py \
  --i /path/to/test_image.mgz \
  --o /path/to/output_segmentation.mgz \
  --checkpoint output/first_training/checkpoints/best_model.pt
```

### Step 3: Verify Output

```bash
# Check if output file exists
ls -lh /path/to/output_segmentation.mgz

# View in FreeView (if available)
freeview /path/to/test_image.mgz /path/to/output_segmentation.mgz
```

### Step 4: Evaluate Prediction

If you have ground truth:

```bash
fspython scripts/freeseg_evaluate.py \
  --gt /path/to/ground_truth.mgz \
  --seg /path/to/output_segmentation.mgz \
  --segmentation_labels output/first_training/segmentation_labels.npy
```

---

## Common Workflows

### Workflow 1: Training from Scratch

```bash
# 1. Prepare data
# 2. Create dataset list
# 3. Configure training
# 4. Train
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --train_output_folder output/training \
  --dataset_list_file data/dataset_list.yaml \
  --dice_epochs 100 \
  --write_tensorboard_summary
```

### Workflow 2: Fine-tuning

```bash
# 1. Start from pre-trained checkpoint
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --checkpoint pretrained_model.pt \
  --train_output_folder output/finetuning \
  --dataset_list_file data/new_dataset_list.yaml \
  --dice_epochs 50 \
  --learning_rate 0.00005  # Lower learning rate
```

### Workflow 3: Batch Prediction

```bash
# 1. Create prediction dataset list
# 2. Run batch prediction
fspython scripts/freeseg_predict.py \
  --dataset_list_file data/prediction_list.yaml \
  --cohort test \
  --o output/predictions/ \
  --checkpoint checkpoints/best_model.pt
```

### Workflow 4: Evaluation

```bash
# 1. Run predictions on test set
# 2. Evaluate all predictions
fspython scripts/freeseg_evaluate.py \
  --gt data/test_labels/ \
  --seg output/predictions/ \
  --segmentation_labels checkpoints/segmentation_labels.npy \
  --path_dice output/dice_scores.npy
```

---

## Next Steps

### Learn More

1. **Read Documentation:**
   - [API Documentation](API.md) - Detailed API reference
   - [Configuration Guide](CONFIGURATION.md) - Configuration details
   - [Training Guide](TRAINING.md) - Training best practices
   - [Architecture Documentation](ARCHITECTURE.md) - System architecture

2. **Explore Examples:**
   - Check `tutorials/` directory for example workflows
   - Review `configs/` for example configurations

3. **Advanced Topics:**
   - Custom augmentations
   - Custom models
   - Multi-GPU training
   - Hyperparameter tuning

### Getting Help

- **Documentation:** See `docs/` directory
- **Issues:** Report issues on GitHub
- **Examples:** Check `tutorials/` directory

---

## Example: Complete Workflow

Here's a complete example workflow:

```bash
# 1. Prepare data (organize in data/ directory)

# 2. Create dataset list
fspython scripts/freeseg_create_data_list.py -d data/ -o dataset_list.yaml

# 3. Configure training
cp configs/config.yaml my_config.yaml
# Edit my_config.yaml with your settings

# 4. Train model
fspython scripts/freeseg_train.py \
  --config my_config.yaml \
  --train_output_folder output/training \
  --dataset_list_file dataset_list.yaml \
  --write_tensorboard_summary \
  --perform_evaluation

# 5. Monitor training
tensorboard --logdir output/training

# 6. Predict on test images
fspython scripts/freeseg_predict.py \
  --dataset_list_file dataset_list.yaml \
  --cohort test \
  --o output/predictions/ \
  --checkpoint output/training/checkpoints/best_model.pt

# 7. Evaluate predictions
fspython scripts/freeseg_evaluate.py \
  --gt data/test_labels/ \
  --seg output/predictions/ \
  --segmentation_labels output/training/segmentation_labels.npy \
  --path_dice output/dice_scores.npy
```

---

Congratulations! You're now ready to use FreeSeg for medical image segmentation. For more detailed information, refer to the other documentation files in the `docs/` directory.

