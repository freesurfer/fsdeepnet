# Configuration Guide

This guide explains how to configure FreeSeg for training, and evaluation during training.

## Table of Contents

- [Configuration File Structure](#configuration-file-structure)
- [Dataset Configuration](#dataset-configuration)
- [Model Configuration](#model-configuration)
- [Preprocessing Configuration](#preprocessing-configuration)
- [Training Configuration](#training-configuration)
- [Augmentation Configuration](#augmentation-configuration)
- [Command-Line Arguments](#command-line-arguments)

---

## Configuration File Structure

FreeSeg uses YAML configuration files to specify all training parameters. The main configuration file is typically named `config.yaml`.

### Basic Structure

```yaml
dataset:
  # Dataset configuration

dataloader:
  # DataLoader configuration

model:
  # Model architecture configuration

preprocessing:
  # Preprocessing and augmentation configuration

training:
  # Training hyperparameters

evaluation:
  # Evaluation configuration
```

---

## Dataset Configuration

### Required Parameters

```yaml
dataset:
  dataset_classname: freeseg.datasets.segmentationdataset.SegmentationDataset
  segmentation_labels: [0, 2, 3, 4, 17, 41, 42, 43, 53]
  expected_num_channels: 1
  dataset_list_file: /path/to/dataset_list.yaml
```

### Optional Parameters

```yaml
dataset:
  # Label names corresponding to segmentation_labels
  segmentation_names: /path/segmentation_names.npy  # or a list
  
  # For left-right symmetric structures (e.g., SynthSeg)
  left_right_corresponding: [2, 41, 3, 42, 4, 43, ...]
  
  # Generation labels (for conditional generation)
  generation_labels: [0, 2, 3, 4]
  generation_classes: [0, 1, 1, 1]
  
  # Topology classes
  topology_classes: [0, 1, 1, 1, ...]
  
  # Parcellation labels
  parcellation_labels: [0, 1000, 1001, ...]
  parcellation_names: /path/parcellation_names.npy
  
  # Resolution difference threshold (default: 0.05 = 5%)
  res_diff_thresh: 0.05
```

### Dataset List File

The `dataset_list_file` should be a YAML file with the following structure:

```yaml
train:
  - image_filepath: /path/train_image1.mgz
    label_filepath: /path/train_label1.mgz
    prior_filepath: /path/train_prior1.mgz  # optional
  - image_filepath: /path/train_image2.mgz
    label_filepath: /path/train_label2.mgz
    prior_filepath: /path/train_prior2.mgz  # optional
    
validation:
  - image_filepath: /path/val_image1.mgz
    label_filepath: /path/val_label1.mgz

test:
  - image_filepath: /path/test_image1.mgz
    label_filepath: /path/test_label1.mgz
```

**Notes:**
- `prior_filepath` is optional and only needed if using prior information
- `image_filepath` is optional if it is used for synthseg styled training
- `image_filepath`, `label_filepath`, and `prior_filepath` should have the same length

---

## Model Configuration

### U-Net Architecture

```yaml
model:
  name: freeseg.models.unet.UNet
  nb_levels: 3                    # Number of U-Net levels (encoder/decoder depth)
  nb_features: 24                 # Base number of features
  feat_mult: 2                    # Feature multiplier per level
  nb_conv_per_level: 2            # Number of convolutions per level
  ndims: 3                        # Number of dimensions (2 or 3)
  conv_size: 3                    # Convolution kernel size
  pool_size: 2                    # Pooling size
  use_residuals: False            # Whether to use residual connections
  refine_conv: False              # Whether to use refinement convolution
  final_pred_activation: softmax  # Final activation function ("softmax", "sigmoid", or "linear")
  weight_init: xavier_uniform     # Weight initialization ("xavier_uniform" or "zeros")
  upsample_interpolation: linear  # Upsample interpolation method ("linear" or "nearest")
  skip_connect: norm              # Where to take the skip connection from ("norm" or "encoder")
  norm: batch                     # Normalization type ("batch" or "instance")
  track_running_stats: False      # Whether to keep running mean and variance
  activation: elu                 # Activation function ("elu" or "relu")
```

### Architecture Constraints

- `crop_size` must be divisible by `2^(nb_levels)`
- For 3D: `crop_size = [W, H, D]`
- For 2D: `crop_size = [W, H]`

**Example:**
- `nb_levels = 3` → `crop_size` must be divisible by 8
- Valid: `[160, 160, 160]`, `[128, 128, 128]`
- Invalid: `[150, 150, 150]` (not divisible by 8)

---

## Preprocessing Configuration

### Basic Preprocessing

```yaml
preprocessing:
  augmentation_class: freeseg.augmentation.augmentbase.AugmentBase
  # Alternative: freeseg.augmentation.augmentvoxynth.AugmentVoxynth
  sampling_hyperparameters: True
  verbose: False
  crop_size: [160, 160, 160]
```

### Augmentation Configuration

Augmentations are specified as a list in the `augmentations` section:

```yaml
preprocessing:
  augmentations:
    - spatialdeformation:
        affine_probability: 1.0
        max_translation: 30      # mm
        max_rotation: 20         # degrees
        max_shearing: 0.015
        max_scaling: 1.2
        warp_probability: 1.0
        warp_integrations: 7
        warp_generation_method: "upsample"  # "blur" or "upsample"
        warp_smoothing_range: [16, 16]
        warp_magnitude_range: [0, 3]
    
    - centercrop:
        max_offset: [1, 2, 3]
    - randomcrop:
    - randomcentercrop:
    - centroidcrop:

    - flip:
        flip_prob: 0.5

    - sampleconditionalgmm:
        prior_mean: [0, 225]
        prior_std: [0, 35]
        prior_distribution: uniform
    
    - biasfieldcorruption:
        bias_field_probability: 1.0
        bias_field_max_magnitude: 0.5
        bias_field_generation_method: "blur"
        bias_field_scale: 0.025
    
    - intensityaugmentation:
        clip_values: [0, 300]
        normalize: True
        added_noise_probability: 1.0
        added_noise_max_sigma: 1.0
        gamma_scaling_probability: 1.0
        gamma_scaling_max: 0.59
    
    - mimicresolution:
        mimic_probability: 1.0
        isotropic_probability: 0.1
        min_res_probability: 0.05
        max_res_iso: 4      # mm
        max_res_aniso: 8    # mm

    - remaplabels:
```

### Available Augmentations

#### 1. Spatial Deformation
- **Purpose**: Applies affine and non-linear spatial transformations
- **Parameters**:
  - `affine_probability`: Probability of applying affine transformation
  - `max_translation`: Maximum translation in mm
  - `max_rotation`: Maximum rotation in degrees
  - `max_shearing`: Maximum shearing
  - `max_scaling`: Maximum scaling factor
  - `warp_probability`: Probability of applying non-linear warp
  - `warp_integrations`: Number of integration steps
  - `warp_generation_method`: Method for displacement field generation ("gaussian" or "perlin")
  - `warp_nonlin_scale`: for warp_generation_method=gaussian
  - `warp_nonlin_std`:  warp_generation_method=gaussian
  - `warp_perlin_method`: Method for displacement field generation when `warp_generation_method` is "perlin" ("upsample", or "blur")
  - `warp_smoothing_range`: Smoothing range for warp field
  - `warp_magnitude_range`: Magnitude range for warp field
- **Note**: Parameters `warp_perlin_method`, `warp_smoothing_range`, `warp_magnitude_range` are for `warp_generation_method` is "perlin".

#### 2. Cropping
- **Types**:
  - `centercrop`: Crop around center with random offsets
  - `randomcrop`: Crop from random location
  - `randomcentercrop`: Crop around random center
  - `centroidcrop`: Crop around label centroid
- **Parameters**:
  - `max_offset`: Maximum offset for center crop [W, H(, D)] (for `centercrop` only)
- **Note**: Only one type of cropping should be listed.

#### 3. Flip
- **Purpose**: Left-right flipping with label swapping
- **Parameters**:
  - `flip_prob`: Probability of flipping (0.0 to 1.0)
- **Note**: Requires `left_right_corresponding` in dataset config

#### 4. Bias Field Corruption
- **Purpose**: Simulates MRI bias field artifacts
- **Parameters**:
  - `bias_field_probability`: Probability of applying bias field corruption
  - `bias_field_max_magnitude`: Maximum value to sample the standard deviation of a normal distribution from U[0, `bias_field_max_magnitude`];
  Maximum bias field magnitude
  - `bias_field_scale`: Ratio between the shape of the input tensor and the shape of the sampled SVF.
  - `bias_field_generation_method`: Method to generate SVF ("blur" or "upsample").
- **Note**: `bias_field_generation_method` is for augmentvoxynth.biasfieldcorruption only.  

#### 5. Intensity Augmentation
- **Purpose**: Applies intensity transformations
- **Parameters**:
  - `clip_values`: Intensity clipping range [min, max]
  - `normalize`: Whether to apply min-max intensities normalization between 0 and 1
  - `added_noise_probability`: Probability of apply noise injection
  - `added_noise_max_sigma`: Maximum value to sample the standard deviation of the Gaussian white noise from U[0, `added_noise_max_sigma`)
  - `gamma_scaling_probability`: Probability of applying gamma augmentation (voxel-wise exponentiation by a randomly sampled power from N(0, `gamma_scaling_max`)
  - `gamma_scaling_max`: Maximum standard deviation the normal distribution from which we sample gamma

#### 6. Mimic Resolution
- **Purpose**: Simulates different image resolutions
- **Parameters**:
  - `mimic_probability`: Probability of applying resolution mimicry
  - `isotropic_probability`: Probability to sample an isotropic resolution if both max_res_iso and max_res_aniso are specified
  - `min_res_probability`: Probability of using original image resolution
  - `max_res_iso`: Maximum isotropic resolution (mm)
  - `max_res_aniso`: Maximum anisotropic resolution (mm)

#### 7. Sample Conditional GMM
- **Purpose**: Conditional intensity image generation using Gaussian Mixture Models
- **Parameters**:
  - `prior_mean`: The means of Gaussian distributions of the GMM
  - `prior_std`: The standard deviations of Gaussian distributions of the GMM
  - `prior_distribution`: Type of distribution ("uniform" or "normal") to sample `prior_mean` and `prior_std`

#### 8. Remap Labels
- **Purpose**: Remaps generation labels to segmentation labels

---

## Training Configuration

### Basic Training Parameters

```yaml
training:
  trainer_class: freeseg.training.Training
  batch_size: 1
  deterministic: False
  report_moving_avg: False
  steps_per_epoch: 1000
  learning_rate: 0.0001
  optimizer: torch.optim.Adam
  write_tensorboard_summary: False
  perform_evaluation: False
  best_model_metric: dice  # "loss" or "dice"
```

### Two-Stage Training

FreeSeg supports two-stage training:

#### Stage 1: Weighted L2 Pre-training

```yaml
training:
  wl2_epochs: 5
  wl2_gt_target_value: 15
  pre_train_learning_rate: 0.0001
  wl2_metrics: freeseg.metrics.WeightedL2Loss
```

- **Purpose**: Pre-train model to predict target values for labels
- **Parameters**:
  - `wl2_epochs`: Number of pre-training epochs
  - `wl2_gt_target_value`: Target value for ground truth labels
  - `pre_train_learning_rate`: Learning rate for pre-training
  - `wl2_metrics`: Loss function class

#### Stage 2: Dice Loss Training

```yaml
training:
  dice_epochs: 100
  dice_squared_form: False
  learning_rate: 0.0001
  model_metrics: freeseg.metrics.DiceLoss
```

- **Purpose**: Fine-tune model using Dice loss
- **Parameters**:
  - `dice_epochs`: Number of training epochs
  - `dice_squared_form`: Whether to use squared Dice form
  - `learning_rate`: Learning rate for training
  - `model_metrics`: Loss function class

### Evaluation During Training

```yaml
training:
  perform_evaluation: True
  best_model_metric: dice  # Metric to select best model
```

When `perform_evaluation=True`, the model is evaluated on the validation set after each epoch, and the best model is saved based on `best_model_metric`.

---

## DataLoader Configuration

```yaml
dataloader:
  num_workers: 0             # Number of data loading workers (0 = data loading occurs in the main process)
  pin_memory: False          # Whether to copy Tensors to CUDA pinned memory.
  persistent_workers: False  # Whether to keep worker processes around
  prefetch_factor: 2         # Number of batches loaded in advance by each worker (ignored if num_workers=0).
                             # This can improve throughput by ensuring workers always have data ready for the main process.
```

**Notes:**
- `num_workers=0`:           Data loading happens in main process (slower but simpler)
- `num_workers>0`:           Parallel data loading (faster but requires more memory).
                             Increasing num_workers can speed up data loading, especially for large datasets, but depends on system resources (CPU cores, I/O speed).
- `pin_memory=True`:         Copy Tensors to CUDA pinned memory. This can accelerate data transfer to the GPU if you are training on a GPU.
- `persistent_workers=True`: The data loader will not shut down the worker processes after each epoch, which can reduce startup overhead for subsequent epochs (only if `num_workers>0`).

---

## Evaluation Configuration

```yaml
evaluation:
  batch_size: 1
```

---

## Command-Line Arguments

Most configuration parameters can be overridden via command-line arguments.

### Training Script Arguments

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --train_output_folder output/training \
  --dataset_list_file data/dataset_list.yaml \
  --crop_size 160 160 160 \
  --batch_size 2 \
  --learning_rate 0.0001 \
  --dice_epochs 100 \
  --wl2_epochs 5 \
  --checkpoint checkpoints/resume.pt \
  --deterministic \
  --write_tensorboard_summary \
  --perform_evaluation \
  --best_model_metric dice \
  --num_workers 4 \
  --pin_memory \
  --cpu \
  --vmp \
  --logfile training.log
```

### Prediction Script Arguments

```bash
fspython scripts/freeseg_predict.py \
  --i input_image.mgz \
  --o output_segmentation.mgz \
  --checkpoint checkpoints/best_model.pt \
  --crop_size 160 160 160 \
  --ctab color_table.ctab \
  --prior input_prior.mgz \
  --gt ground_truth.mgz \
  --write_posteriors \
  --cpu \
  --logfile prediction.log
```

### Evaluation Script Arguments

```bash
fspython scripts/freeseg_evaluate.py \
  --gt ground_truth_folder \
  --seg segmentation_folder \
  --segmentation_labels segmentation_labels.npy \
  --evaluation_labels 2 3 4 17 41 \
  --path_dice dice_scores.npy \
  --logfile evaluation.log
```

---

## Configuration Best Practices

1. **Start Simple**: Begin with minimal augmentations and gradually add more
2. **Crop Size**: Choose crop sizes divisible by `2^(nb_levels)`
3. **Batch Size**: Adjust based on GPU memory (typically 1-4 for 3D)
4. **Learning Rate**: Start with 0.0001 and adjust based on training dynamics
5. **Augmentation Probability**: Use 1.0 for training, 0.0 for validation
6. **Two-Stage Training**: Use weighted L2 pre-training for better initialization
7. **Evaluation**: Enable `perform_evaluation` to track validation performance
8. **TensorBoard**: Enable `write_tensorboard_summary` for visualization

---

## Example Configuration Files

See `configs/config.yaml` for a complete example configuration file.

