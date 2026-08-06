# Getting Started with Fsdeepnet

This guide will help you get started with Fsdeepnet.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Common Workflow](#common-workflows)
- [Next Steps](#next-steps)

---

## Installation

### Prerequisites

1. **FreeSurfer**: Fsdeepnet requires FreeSurfer to be installed and configured
2. **Python**: Python 3.8 (via FreeSurfer's fspython)
3. **CUDA**: GPU with CUDA support (optional but recommended)

### Installation Steps

1. **Build FreeSurfer with Python and CUDA support:**
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
   git clone https://github.com/freesurfer/fsdeepnet.git
   cd fsdeepnet
   ```

4. **Install Fsdeepnet:**
   ```bash
   # Standard installation
   fspython -m pip install .
   
   # Editable installation (for development)
   fspython -m pip install --editable .
   ```

---

## Quick Start

### 1. Prepare Your Dataset List

- Method 1: Manually prepare a YAML file (`dataset_list.yaml`):

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

- Method 2: Use the helper script `scripts/fsdeepnet_create_data_list.py`:

  - #### **Organize data:** Organize your data in the following structure:

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

  - #### **Create dataset list:** Use the helper script
  ```bash
   fspython scripts/fsdeepnet_split_dataset.py \
     -d datafolder/ \
     -o dataset_list.yaml
  ```

### 2. Create Configuration File

Copy the example configuration:

```bash
cp configs/config.yaml my_config.yaml
```

Edit `my_config.yaml`:

```yaml
dataset:
  class_name: fsdeepnet.datasets.segmentationdataset.SegmentationDataset
  segmentation_labels: [0, 2, 3, 4, 17, 41]  # Your label values
  expected_num_channels: 1
  dataset_list_file: /path/to/dataset_list.yaml

model:
  name: fsdeepnet.models.unet.UNet
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

### 3. Train Your Model

```bash
fspython scripts/fsdeepnet_train.py \
  --config my_config.yaml \
  --dataset_list_file /path/to/dataset_list.yaml \  
  --train_output_folder output/my_training
```

### 4. Predict on New Images

```bash
fspython scripts/fsdeepnet_predict.py \
  --i /path/to/new_image.mgz \
  --o /path/to/output_segmentation.mgz \
  --checkpoint output/my_training/checkpoints/best_model.pth
```

---

## Common Workflows

### Workflow 1: Training from Scratch

```bash
# 1. Prepare data
# 2. Create dataset list
# 3. Configure training
# 4. Train
fspython scripts/fsdeepnet_train.py \
  --config configs/config.yaml \
  --dataset_list_file data/dataset_list.yaml \  
  --train_output_folder output/training \
  --dice_epochs 100
```

### Workflow 2: Fine-tuning

```bash
# 1. Start from pre-trained checkpoint
fspython scripts/fsdeepnet_train.py \
  --config configs/config.yaml \
  --dataset_list_file data/new_dataset_list.yaml \  
  --train_output_folder output/finetuning \
  --checkpoint pretrained_model.pth \  # pre-trained model 
  --dice_epochs 50 \  # 50 epochs more
  --learning_rate 0.00005  # lower learning rate
```

### Workflow 3: Batch Prediction

```bash
# 1. Create prediction dataset list
# 2. Run batch prediction
fspython scripts/fsdeepnet_predict.py \
  --dataset_list_file data/prediction_list.yaml \
  --cohort test \
  --o output/predictions/ \
  --checkpoint checkpoints/best_model.pth
```

### Workflow 4: Evaluation

```bash
# 1. Run predictions on test set
# 2. Evaluate all predictions
fspython scripts/fsdeepnet_evaluate.py \
  --gt data/test_labels/ \
  --seg output/predictions/ \
  --segmentation_labels checkpoints/segmentation_labels.npy \
  --path_dice output/dice_scores.npy
```

---

## Next Steps

### Learn More

1. **Read Documentation:**
   - [API Documentation (WIP)](API.md) - Detailed API reference
   - [Configuration Guide](CONFIGURATION.md) - Configuration details
   - [Training Guide](TRAINING.md) - Training options
   - [Architecture Documentation](ARCHITECTURE.md) - System architecture

2. **Explore Examples:**
   - Check `tutorials/` directory for example workflows
   - Review `configs/` for example configurations

### Getting Help

- **Documentation:** See `docs/` directory
- **Issues:** Report issues on GitHub
- **Examples:** Check `tutorials/` directory

---

Congratulations! You're now ready to use Fsdeepnet. For more detailed information, refer to the other documentations in the `docs/` directory.

