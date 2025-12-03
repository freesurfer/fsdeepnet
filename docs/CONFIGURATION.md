# Configuration Guide

This guide explains how to configure FreeSeg for training.

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

FreeSeg uses YAML configuration files to specify all training parameters.

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

### Example Configuration Files

See `configs/config.yaml` and `configs/synthseg_config.yaml` for complete example configuration files.

---

## Dataset Configuration

### Required Parameters

```yaml
dataset:
  # torch.utils.data.Dataset class
  dataset_classname: freeseg.datasets.segmentationdataset.SegmentationDataset
  
  # label ids in the final segmentation 
  segmentation_labels: [0, 2, 3, 4, 17, 41, 42, 43, 53]

  # number of channels of the input data
  expected_num_channels: 1

  # the dataset list YAML file
  dataset_list_file: /path/to/dataset_list.yaml
```

### Optional Parameters

```yaml
dataset:
  # label names corresponding to segmentation_labels
  segmentation_names: /path/segmentation_names.npy, or a list
  
  # left-right corresponding structures
  # required for `flip` augmentation
  left_right_corresponding: list 
  
  # all possible label ids in the training label maps
  # needs to be the same length as `segmentation_labels`
  # all occurrences of generation_labels[i] in the input label maps will be mapped to segmentation_labels[i]
  # required for `remaplabels` augmentation
  generation_labels: /path/generation_labels.npy, or a list

  # indices regrouping `segmentation_labels` into classes of same intensity distribution
  # labels regrouped to the same class will share the same Gaussian when sampling a new intensity image using Gaussian Mixture Models
  generation_classes: /path/generation_classes.npy, or a list
  
  # Topology classes
  topology_classes: /path/topology_classes.npy, or a list
  
  # parcellation labels and names
  parcellation_labels: /path/parcellation_labels.npy, or a list
  parcellation_names: /path/parcellation_names.npy, or a list
  
  # resolution difference threshold (default: 0.05 = 5%)
  # the input image will be resampled to target resolution if its resolution is not within `target resolution+-res_diff_thresh`
  # the target resolution is obtained from the training data
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
- `image_filepath`, `label_filepath`, and `prior_filepath` should have the same number of entries

---

## Model Configuration

### U-Net Architecture

```yaml
model:
  name: freeseg.models.unet.UNet   # torch.nn.Module class
  nb_levels: 3                     # number of U-Net levels (encoder/decoder depth)
  nb_features: 24                  # base number of features
  feat_mult: 2                     # feature multiplier per level
  nb_conv_per_level: 2             # number of convolutions per level
  ndims: 3                         # number of dimensions (2 or 3)
  conv_size: 3                     # convolution kernel size
  pool_size: 2                     # pooling size
  use_residuals: False             # whether to use residual connections
  refine_conv: False               # whether to use refinement convolution
  final_pred_activation: softmax   # final activation function ("softmax", "sigmoid", or "linear")
  weight_init: xavier_uniform      # weight initialization ("xavier_uniform" or "zeros")
  upsample_interpolation: linear   # upsample interpolation method ("linear" or "nearest")
  skip_connect: norm               # where to take the skip connection from ("norm" or "encoder")
  norm: batch                      # normalization type ("batch" or "instance")
  track_running_stats: False       # whether to keep running mean and variance
  activation: elu                  # activation function ("elu" or "relu")
```

---

## Preprocessing Configuration

### Basic Preprocessing

```yaml
preprocessing:
  # augmentation wrapper class
  # alternative: freeseg.augmentation.augmentvoxynth.AugmentVoxynth
  augmentation_class: freeseg.augmentation.augmentbase.AugmentBase
  
  # constrained by the U-Net architecture
  # must be divisible by `2^(nb_levels)`
  crop_size: [W, H(, D)]
```

### Augmentation Configuration

Augmentations are specified as a list in the `augmentations` section along with the hyperparameters. They are applied in the same order as specified. The augmentation names are case-insensitive.

### Available Augmentations

#### 1. **`spatialdeformation`**
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

#### 2. **cropping methods**
- **Choices**:
  - `centercrop`: Crop around center with random offsets sampled from U[`-max_offset`, `+max_offset`]
  - `randomcrop`: Crop from random location
  - `randomcentercrop`: Crop around random center
  - `centroidcrop`: Crop around label centroid with random offsets sampled from U[`-max_offset`, `+max_offset`]
- **Parameters**:
  - `max_offset`: For `centercrop` and `centroidcrop` only, maximum value [W, H(, D)] to sample a random offset from U[`-max_offset`, `+max_offset`]
- **Note**: Only one cropping method should be listed.

#### 3. **`flip`**
- **Purpose**: Left-right flipping with label swapping
- **Parameters**:
  - `flip_prob`: Probability of flipping (0.0 to 1.0)
- **Note**: Requires `left_right_corresponding` in dataset config

#### 4. **`biasfieldcorruption`**
- **Purpose**: Simulates MRI bias field artifacts
- **Parameters**:
  - `bias_field_probability`: Probability of applying bias field corruption
  - `bias_field_max_magnitude`: Maximum value to sample the standard deviation of a normal distribution from U[0, `bias_field_max_magnitude`];
  - `bias_field_scale`: Ratio between the shape of the input tensor and the shape of the sampled SVF.
  - `bias_field_generation_method`: Method to generate SVF ("blur" or "upsample").
- **Note**: `bias_field_generation_method` is for augmentvoxynth.biasfieldcorruption only.  

#### 5. **`intensityaugmentation`**
- **Purpose**: Applies intensity transformations
- **Parameters**:
  - `clip_values`: Intensity clipping range [min, max]
  - `normalize`: Whether to apply min-max intensities normalization between 0 and 1
  - `added_noise_probability`: Probability of apply noise injection
  - `added_noise_max_sigma`: Maximum value to sample the standard deviation of the Gaussian white noise from U[0, `added_noise_max_sigma`)
  - `gamma_scaling_probability`: Probability of applying gamma augmentation (voxel-wise exponentiation by a randomly sampled power from N(0, `gamma_scaling_max`)
  - `gamma_scaling_max`: Maximum standard deviation the normal distribution from which we sample gamma

#### 6. **`mimicresolution`**
- **Purpose**: Simulates different image resolutions
- **Parameters**:
  - `mimic_probability`: Probability of applying resolution mimicry
  - `isotropic_probability`: Probability to sample an isotropic resolution if both max_res_iso and max_res_aniso are specified
  - `min_res_probability`: Probability of using original image resolution
  - `max_res_iso`: Maximum isotropic resolution (mm)
  - `max_res_aniso`: Maximum anisotropic resolution (mm)

#### 7. **`sampleconditionalgmm`**
- **Purpose**: Conditional intensity image generation using Gaussian Mixture Models
- **Parameters**:
  - `prior_mean`: The Gaussian means of the GMM
  - `prior_std`: The Gaussian standard deviations of the GMM
  - `prior_distribution`: Type of distribution ("uniform" or "normal") to sample `prior_mean` and `prior_std`
- **Note**: Requires `generation_classes` in dataset config
  

#### 8. **`remaplabels`**
- **Purpose**: Remaps `generation labels` to `segmentation labels`
- **Note**: Requires `generation_labels` in dataset config


### Example:
```yaml
preprocessing:
  augmentations:
    # Spatial Deformation
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

    # choices of cropping methods: `centercrop`, `randomcrop`, `randomcentercrop`, and `centroidcrop`
    - centercrop:
        max_offset: [1, 2, 3]

    # left-right flipping with label swapping
    - flip:
        flip_prob: 0.5

    # Sample Conditional GMM
    - sampleconditionalgmm:
        prior_mean: [0, 225]
        prior_std: [0, 35]
        prior_distribution: uniform

    # Bias Field Corruption
    - biasfieldcorruption:
        bias_field_probability: 1.0
        bias_field_max_magnitude: 0.5
        bias_field_generation_method: "blur"
        bias_field_scale: 0.025

    # Intensity Augmentation
    - intensityaugmentation:
        clip_values: [0, 300]
        normalize: True
        added_noise_probability: 1.0
        added_noise_max_sigma: 1.0
        gamma_scaling_probability: 1.0
        gamma_scaling_max: 0.59

    # Mimic Resolution
    - mimicresolution:
        mimic_probability: 1.0
        isotropic_probability: 0.1
        min_res_probability: 0.05
        max_res_iso: 4      # mm
        max_res_aniso: 8    # mm

    # Remap Labels
    - remaplabels:
```

---

## Training Configuration

### Basic Training Parameters

```yaml
training:
  trainer_class: freeseg.training.Training   # the trainer class
  batch_size: 1                              # number of training samples passed through network per training step
  deterministic: False                       # whether to do deterministic training
  report_moving_avg: False                   # whether to report moving average training loss and dice
  steps_per_epoch: 1000                      # number of steps per training epoch
  optimizer: torch.optim.Adam                # optimizer class
  write_tensorboard_summary: False           # whether to write tensorboard summary (to be tested)
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

- **Purpose**: Pre-train model with weighted L2 norm loss function to provide initialization for Dice loss training
- **Parameters**:
  - `wl2_epochs`: Number of pre-training epochs
  - `wl2_gt_target_value`: Target value for ground truth labels of the layer before `final_pred_activation`: gt_target_value when gt = 1, -gt_target_value when gt = 0.
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

- **Purpose**: Fine-tune model using soft Dice loss function
- **Parameters**:
  - `dice_epochs`: Number of training epochs
  - `dice_squared_form`: Whether to use squared Dice form
  - `learning_rate`: Learning rate for training
  - `model_metrics`: Loss function class

### Perform Evaluation During Training

```yaml
training:
  perform_evaluation: False                  # whether to perform evaluation at the end of each training epoch
  best_model_metric: dice                    # metric to pick best model when `perform_evaluation=True` ("loss" or "dice")
```

When `perform_evaluation=True`, the model is evaluated on the validation set at the end of each training epoch, and the best model is saved based on `best_model_metric`.

---

## DataLoader Configuration

```yaml
dataloader:
  num_workers: 0              # Number of data loading workers (0 = data loading occurs in the main process)
  pin_memory: False           # Whether to copy Tensors to CUDA pinned memory.
  persistent_workers: False   # Whether to keep worker processes around
  prefetch_factor: 2          # Number of batches loaded in advance by each worker (ignored if num_workers=0).
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

## Training Script Command-Line Arguments

Most configuration parameters can be overridden via `scripts/freeseg_train.py` command-line arguments.

**Example:**

```bash
fspython scripts/freeseg_train.py \
  --config configs/config.yaml \
  --dataset_list_file data/dataset_list.yaml \
  --train_output_folder output/training \
  --crop_size 160 160 160 \
  --batch_size 2 \
  --learning_rate 0.0001 \
  --dice_epochs 100 \
  --wl2_epochs 5 \
  --checkpoint checkpoints/resume.pth \
  --deterministic \
  --perform_evaluation \
  --best_model_metric dice \
  --num_workers 4 \
  --pin_memory \
  --cpu \
  --vmp \
  --logfile training.log
```


