# Architecture Documentation

This document describes the architecture and design of FreeSeg.

## Table of Contents

- [System Overview](#system-overview)
- [U-Net Architecture](#u-net-architecture)
- [Data Pipeline](#data-pipeline)
- [Training Pipeline](#training-pipeline)
- [Inference Pipeline](#inference-pipeline)
- [Module Structure](#module-structure)

---

## System Overview

FreeSeg is a PyTorch-based deep learning framework for medical image segmentation, specifically designed for FreeSurfer-adjacent models. The system is modular and extensible, supporting various architectures, augmentations, and training strategies.

### Key Components

1. **Models**: Neural network architectures (U-Net)
2. **Datasets**: Data loading and preprocessing
3. **Augmentation**: Data augmentation pipeline
4. **Training**: Training loop and optimization
5. **Prediction**: Inference and segmentation
6. **Evaluation**: Metrics and evaluation tools
7. **Configuration**: Configuration management

---

## U-Net Architecture

### Architecture Overview

FreeSeg implements a 3D/2D U-Net architecture with the following features:

- **Encoder-Decoder Structure**: Symmetric encoder and decoder paths
- **Skip Connections**: Feature concatenation between encoder and decoder
- **Multi-Scale Features**: Hierarchical feature extraction
- **Flexible Depth**: Configurable number of levels
- **Residual Connections**: Optional residual blocks
- **Normalization**: Batch or instance normalization

### Architecture Diagram

```
Input [N, C, H, W, D]
  |
  v
Level 0 (Encoder)
  |-- ConvBlock (nb_features)
  |-- Downsample (pool_size)
  |
  v
Level 1 (Encoder)
  |-- ConvBlock (nb_features * feat_mult)
  |-- Downsample
  |
  v
Level 2 (Encoder)
  |-- ConvBlock (nb_features * feat_mult^2)
  |-- Downsample
  |
  v
Bottleneck
  |-- ConvBlock (nb_features * feat_mult^nb_levels)
  |
  v
Level 2 (Decoder)
  |-- Upsample
  |-- Skip Connection (from Level 2 Encoder)
  |-- ConvBlock
  |
  v
Level 1 (Decoder)
  |-- Upsample
  |-- Skip Connection (from Level 1 Encoder)
  |-- ConvBlock
  |
  v
Level 0 (Decoder)
  |-- Upsample
  |-- Skip Connection (from Level 0 Encoder)
  |-- ConvBlock
  |
  v
Output [N, nb_labels, H, W, D]
```

### Architecture Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `nb_levels` | Number of encoder/decoder levels | 3 |
| `nb_features` | Base number of features | 24 |
| `feat_mult` | Feature multiplier per level | 2 |
| `nb_conv_per_level` | Convolutions per level | 2 |
| `conv_size` | Convolution kernel size | 3 |
| `pool_size` | Pooling/downsampling size | 2 |
| `use_residuals` | Use residual connections | False |
| `ndims` | Number of dimensions (2 or 3) | 3 |
| `norm` | Normalization type | "batch" |
| `activation` | Activation function | "elu" |

### Feature Progression

For a U-Net with `nb_levels=3`, `nb_features=24`, `feat_mult=2`:

- **Level 0**: 24 features
- **Level 1**: 48 features (24 × 2)
- **Level 2**: 96 features (24 × 2²)
- **Bottleneck**: 192 features (24 × 2³)

### Convolutional Block

Each level contains `nb_conv_per_level` convolutional blocks:

```
Input
  |
  v
Conv (in_channels → out_channels)
  |
  v
Activation (ELU/ReLU)
  |
  v
[Repeat nb_conv_per_level times]
  |
  v
[Optional: Residual Connection]
  |
  v
Output
```

### Skip Connections

Skip connections concatenate encoder features with decoder features:

- **Type**: Feature concatenation (not addition)
- **Normalization**: Optional normalization before concatenation
- **Connection**: From encoder level N to decoder level N

### Output Layer

- **Final Convolution**: `nb_features → nb_labels`
- **Activation**: Softmax (multi-class) or Sigmoid (binary)
- **Output Shape**: `[N, nb_labels, H, W, D]`

---

## Data Pipeline

### Data Flow

```
Raw Images/Labels
  |
  v
Dataset List (YAML)
  |
  v
SegmentationDataset
  |-- Load images/labels
  |-- Apply augmentations
  |-- Crop to crop_size
  |-- Normalize
  |
  v
DataLoader
  |-- Batch creation
  |-- Shuffling
  |-- Multi-worker loading
  |
  v
Training Loop
```

### Dataset Class

**SegmentationDataset** (`freeseg.datasets.segmentationdataset`)

- **Input**: Dataset list with image/label/prior filepaths
- **Processing**:
  - Load medical images (MGZ, NIFTI, etc.)
  - Apply augmentations
  - Crop to specified size
  - Convert to tensors
- **Output**: Batched tensors for training

### Augmentation Pipeline

**AugmentBase** (`freeseg.augmentation.augmentbase`)

Augmentations are applied in sequence:

1. **Spatial Deformation**: Affine + non-linear warping
2. **Cropping**: Center/random/centroid cropping
3. **Flipping**: Left-right flipping (with label swapping)
4. **Bias Field**: MRI bias field corruption
5. **Intensity**: Noise, gamma correction, normalization
6. **Resolution**: Resolution mimicking
7. **Label Remapping**: Label remapping

### Data Loading

- **Workers**: Multi-process data loading
- **Prefetching**: Prefetch batches for faster training
- **Memory Pinning**: Pin memory for GPU transfer
- **Persistent Workers**: Keep workers alive between epochs

---

## Training Pipeline

### Training Flow

```
Configuration (YAML + CLI args)
  |
  v
Config Processing
  |-- Load YAML
  |-- Override with CLI args
  |-- Validate parameters
  |
  v
Model Setup
  |-- Initialize model
  |-- Load checkpoint (if resuming)
  |-- Move to device
  |
  v
Dataset Setup
  |-- Load dataset list
  |-- Create train/val datasets
  |-- Create data loaders
  |
  v
Training Loop
  |-- Stage 1: Weighted L2 (if enabled)
  |-- Stage 2: Dice Loss
  |-- Epoch loop
  |   |-- Batch loop
  |   |   |-- Forward pass
  |   |   |-- Loss computation
  |   |   |-- Backward pass
  |   |   |-- Optimizer step
  |   |-- Validation (if enabled)
  |   |-- Checkpoint saving
  |
  v
Best Model Selection
```

### Two-Stage Training

#### Stage 1: Weighted L2 Pre-training

- **Purpose**: Initialize model with target value prediction
- **Loss**: Weighted L2 loss
- **Target**: Predict `wl2_gt_target_value` for ground truth labels
- **Duration**: `wl2_epochs` epochs

#### Stage 2: Dice Loss Training

- **Purpose**: Fine-tune for segmentation
- **Loss**: Dice loss (soft or hard)
- **Optimization**: Adam optimizer
- **Duration**: `dice_epochs` epochs

### Training Components

**Training Class** (`freeseg.training.Training`)

- **Model Management**: Model initialization, checkpointing
- **Optimization**: Optimizer setup, learning rate scheduling
- **Loss Computation**: Dice loss, weighted L2 loss
- **Metrics**: Dice scores, loss tracking
- **Evaluation**: Validation during training
- **Logging**: TensorBoard, log files

### Checkpoint System

**Checkpoint Class** (`freeseg.checkpoint.Checkpoint`)

Checkpoints contain:
- Model state dictionary
- Optimizer state dictionary
- Training epoch
- Best metrics (loss, dice)
- Model architecture
- Dataset configuration
- Label lookup table

---

## Inference Pipeline

### Prediction Flow

```
Input Image
  |
  v
Model Loading
  |-- Load checkpoint
  |-- Initialize model
  |-- Load weights
  |
  v
Preprocessing
  |-- Load image
  |-- Resample (if needed)
  |-- Normalize
  |-- Crop (if needed)
  |
  v
Inference
  |-- Forward pass
  |-- Softmax/Sigmoid
  |-- Argmax (for labels)
  |
  v
Post-processing
  |-- Smooth posteriors (optional)
  |-- Left-right flipping (optional)
  |-- Combine predictions
  |
  v
Output Segmentation
```

### Prediction Class

**Prediction Class** (`freeseg.prediction.Prediction`)

- **Model Building**: Load and assemble models
- **Inference**: Run predictions on images
- **Post-processing**: Smoothing, flipping, combining
- **Output**: Save segmentations and posteriors

### Inference Modes

1. **Single Image**: Predict on one image
2. **Batch**: Predict on multiple images
3. **With Priors**: Use prior information
4. **With Flipping**: Test-time augmentation
5. **Smooth Posteriors**: Gaussian smoothing

---

## Module Structure

### Core Modules

```
freeseg/
  |-- __init__.py          # Package initialization
  |-- config.py            # Configuration management
  |-- training.py          # Training class
  |-- prediction.py        # Prediction class
  |-- evaluation.py        # Evaluation class
  |-- checkpoint.py         # Checkpoint management
  |-- metrics.py           # Loss functions and metrics
  |-- filter.py            # Filtering utilities
```

### Model Modules

```
freeseg/models/
  |-- __init__.py
  |-- unet.py              # U-Net architecture
```

### Dataset Modules

```
freeseg/datasets/
  |-- __init__.py
  |-- segmentationdataset.py  # Segmentation dataset
```

### Augmentation Modules

```
freeseg/augmentation/
  |-- __init__.py
  |-- augmentbase.py      # Base augmentation class
  |-- augmentvoxynth.py   # Voxynth augmentation class
```

### Utility Modules

```
freeseg/utils/
  |-- __init__.py
  |-- utility.py          # Utility functions
```

### Voxynth Modules

```
freeseg/voxynth/
  |-- __init__.py
  |-- augment.py          # Voxynth augmentations
  |-- filter.py           # Filtering
  |-- noise.py            # Noise generation
  |-- synth.py            # Synthesis
  |-- transform.py        # Transformations
  |-- utility.py          # Utilities
```

---

## Design Principles

### 1. Modularity

- Each component is independent and reusable
- Clear separation of concerns
- Easy to extend and modify

### 2. Flexibility

- Configurable via YAML and CLI
- Support for different architectures
- Extensible augmentation pipeline

### 3. Reproducibility

- Deterministic training option
- Checkpoint saving/loading
- Configuration saving

### 4. Performance

- Efficient data loading
- GPU acceleration
- Memory optimization

### 5. Usability

- Simple command-line interface
- Comprehensive logging
- TensorBoard integration

---

## Extension Points

### Adding New Models

1. Create model class in `freeseg/models/`
2. Implement `__init__` and `forward` methods
3. Register in configuration

### Adding New Augmentations

1. Create augmentation class in `freeseg/augmentation/`
2. Inherit from base augmentation class
3. Implement `forward` method
4. Add to valid augmentations list

### Adding New Metrics

1. Create metric class in `freeseg/metrics.py`
2. Inherit from `nn.Module`
3. Implement `forward` method
4. Register in configuration

### Adding New Datasets

1. Create dataset class in `freeseg/datasets/`
2. Inherit from `torch.utils.data.Dataset`
3. Implement `__getitem__` and `__len__`
4. Register in configuration

---

## Performance Considerations

### Memory Optimization

- **Crop Size**: Smaller crops use less memory
- **Batch Size**: Reduce for large images
- **Gradient Accumulation**: Simulate larger batches
- **Mixed Precision**: Use FP16 for faster training

### Speed Optimization

- **Data Loading**: Multi-worker loading
- **Prefetching**: Prefetch batches
- **Memory Pinning**: Faster GPU transfer
- **Persistent Workers**: Avoid worker recreation

### GPU Utilization

- **Batch Size**: Maximize GPU memory usage
- **Crop Size**: Balance memory and context
- **Multi-GPU**: DataParallel or DistributedDataParallel

---

## Future Enhancements

Potential improvements:

1. **Additional Architectures**: V-Net, Attention U-Net
2. **Advanced Augmentations**: Mixup, CutMix
3. **Loss Functions**: Focal loss, Tversky loss
4. **Optimization**: Learning rate scheduling, warmup
5. **Distributed Training**: Multi-GPU, multi-node
6. **Model Compression**: Quantization, pruning

---

## References

- U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
- PyTorch Documentation: https://pytorch.org/docs/
- FreeSurfer: https://freesurfer.net/

