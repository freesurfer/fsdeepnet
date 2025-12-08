# Architecture Documentation

This document describes the architecture and design of FreeSeg.

## Table of Contents

- [System Overview](#system-overview)
- [U-Net Architecture](#u-net-architecture)
- [Augmentation Pipeline](#augmentation-pipeline)
- [Training Pipeline](#training-pipeline)
- [Inference Pipeline](#inference-pipeline)
- [Extension Points](#extension-points)
- [Future Enhancements](#future-enhancements)
- [References](#references)

---

## System Overview

FreeSeg is a PyTorch-based deep learning framework, specifically designed for FreeSurfer-adjacent models. The system is modular and extensible, supporting various architectures, augmentations, and training strategies.

### Key Components

- **Models**: Neural network architectures (U-Net)
- **Datasets**: Data loading and preprocessing
- **Augmentation**: Data augmentation pipeline
- **Training**: Training loop and optimization
- **Prediction**: Inference and segmentation
- **Evaluation**: Metrics and evaluation tools
- **Configuration**: Configuration management

\
\
![system workflow diagram](figures/system_workflow.png)

### Module Structure

#### Core Modules

```
freeseg/
  |-- __init__.py          # Package initialization
  |-- config.py            # Configuration management
  |-- training.py          # Training class
  |-- prediction.py        # Prediction class
  |-- evaluation.py        # Evaluation class
  |-- checkpoint.py        # Checkpoint management
  |-- metrics.py           # Loss functions and metrics
  |-- filter.py            # Filtering utilities
```

#### Model Modules

```
freeseg/models/
  |-- __init__.py
  |-- unet.py              # U-Net architecture
```

#### Dataset Modules

```
freeseg/datasets/
  |-- __init__.py
  |-- segmentationdataset.py  # Segmentation dataset
```

#### Augmentation Modules

```
freeseg/augmentation/
  |-- __init__.py
  |-- augmentbase.py      # Base augmentation class
  |-- augmentvoxynth.py   # Voxynth augmentation class
```

#### Utility Modules

```
freeseg/utils/
  |-- __init__.py
  |-- utility.py          # Utility functions
```

#### Voxynth Modules
**Note**: Modified from Voxynth implementation https://github.com/dalcalab/voxynth/

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

### Design Principles

#### Modularity

- Each component is independent and reusable
- Easy to extend and modify

#### Flexibility

- Configurable via YAML and CLI
- Extensible components (network architectures, augmentation pipeline, ...)

#### Reproducibility

- Deterministic training option
- Checkpoint saving/loading
- Configuration saving

#### Usability

- Simple command-line interface
- Comprehensive logging
- TensorBoard integration (to be tested)
\
\
![design principle diagram](figures/design_principle.png)

---

## U-Net Architecture

### Overview

FreeSeg implements a 3D/2D U-Net architecture with the following features:

- **Encoder-Decoder Structure**: Symmetric encoder and decoder paths
- **Skip Connections**: Feature concatenation (not addition) between encoder and decoder
- **Multi-Scale Features**: Hierarchical feature extraction
- **Flexible Depth**: Configurable number of levels
- **Residual Connections**: Optional residual blocks
- **Normalization**: Batch or instance normalization

### Network Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ndims` | Number of dimensions (2 or 3) | 3 |
| `nb_levels` | Number of encoder/decoder levels | 3 |
| `nb_features` | Base number of features | 24 |
| `feat_mult` | Feature multiplier per level | 2 |
| `nb_conv_per_level` | Convolutions per level | 2 |
| `conv_size` | Convolution kernel size | 3 |
| `pool_size` | Pooling/downsampling size | 2 |
| `use_residuals` | Use residual connections (False or True) | False |
| `norm` | Normalization type ("batch" or "instance") | "batch" |
| `track_running_stats` | Keep running mean and variance (False or True) | False |
| `activation` | Activation function ("elu" or "relu") | "elu" |
| `final_pred_activation` | Final activation function ("softmax", "sigmod", or "linear") | "softmax" |
| `upsample_interpolation` | Upsample interpolation method ("linear" or "nearest") | "linear" |
| `weight_init` | Weight initialization ("xavier_uniform" or "zeros") | "xavier_uniform" |
| `skip_connect` | Where to take the skip connection from ("norm" or "encoder") | "norm" |

### Network Diagram

Example U-Net with `ndim=3`, `nb_levels=5`, `nb_features=24`, `nb_conv_per_level=2`, `feat_mult=2`, `conv_size=3`, `pool_size=2`, `norm=batch`, `activation=elu`, `final_pred_activation=softmax`:
\
\
![unet diagram](figures/unet_diagram.png)

### Output Layer

- **Final Convolution**: `nb_features → nb_labels`
- **Activation**: Softmax (multi-class) or Sigmoid (binary)
- **Output Shape**: `[N, nb_labels, H, W, D]`

---

## Augmentation Pipeline

Augmentations are applied in the order they are specified in the configuration file.

- **`freeseg.augmentation.augmentbase`**
  - **`AugmentBase`**: base augmentation wrapper class
  - **Individual augmentation classes**
    - **`SpatialDeformation`**: Affine + non-linear spatial transformations
    - **Cropping classes**: choose from one of the follow
      - **`CenterCrop`**
      - **`RandomCrop`**
      - **`CentroidCrop`**
    - **`Flip`**: Left-right flipping (with label swapping)
    - **`BiasFieldCorruption`**: MRI bias field corruption
    - **`IntensityAugmentation`**: Noise, gamma correction, normalization
    - **`MimicResolution`**: Resolution mimicking
    - **`RemapLabels`**: Label remapping
    - **`SampleConditionalGMM`**: Conditional intensity image generation using Gaussian Mixture Models
- **`freeseg.augmentation.augmentvoxynth`**
  - **`AugmentVoxynth`**: derived augementation wrapper class `AugmentVoxynth` → `AugmentBase`
  - **Individaul augmentation classes** (implemented using Voxynth library https://github.com/dalcalab/voxynth/)
    - **`BiasFieldCorruption`**: MRI bias field corruption
    - **`IntensityAugmentation`**: Noise, gamma correction, normalization

---

## Training Pipeline

### Two-Stage Training

#### Stage 1: Weighted L2 Pre-training

- **Purpose**: Pre-train model with weighted L2 norm loss function to provide initialization for Dice loss training
- **Loss**: Weighted L2 Loss
- **Duration**: `wl2_epochs` epochs

#### Stage 2: Dice Loss Training

- **Purpose**: Fine-tune model using soft Dice loss
- **Loss**: Soft Dice Loss
- **Duration**: `dice_epochs` epochs

### Training Components

**Training Class** (`freeseg.training.Training`)

- **Model Management**: Model initialization, checkpointing
- **Optimization**: Optimizer setup
- **Loss Computation**: Dice loss, weighted L2 loss
- **Metrics**: Dice scores, loss tracking
- **Evaluation**: Validation during training
- **Logging**: log files, TensorBoard (??? to be tested)

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
Inference Model Building
  |-- Load checkpoint
  |-- Initialize model
  |-- Load weights
  |-- Assemble inference model
  |
  v
Preprocessing
  |-- Load image
  |-- Resample image to target resolution (if needed)
  |-- Crop image (if needed)
  |-- Normalize image
  |-- Pad image (if needed)
  |
  v
Inference
  |-- Inference model forward pass
  |
  v
Postprocessing
  |-- Remove posteriors padding (if needed)
  |-- Set posteriors outside the biggest connected component to zero (optional)
  |-- Set posteriors outside the largest connected component of each topological class to zero (optional)
  |-- Get hard segmentation
  |-- Combine segmentation and parcellation (optional)
  |
  v
Output Segmentation
```

### Prediction Class

**Prediction Class** (`freeseg.prediction.Prediction`)

- **`__init__`**: Class constructor
- **`build_model`**: Load and assemble models
  ```
  inference model =   segmentation model
                    + smooth posteriors (optional, '--smooth_posteriors')
                    + left-right flipped image prediction (optional, '--flip')
                    + parcellation model (optional)
  ```
- **`predict`**: Predict with the assembled inference model
  ```
  |-- Preprocess
  |   |-- Load image
  |   |-- Resample image to target resolution (if needed)
  |   |-- Crop image (if needed)
  |   |-- Normalize image
  |   |-- Pad image (if needed)
  |-- Run images through the inference model
  |-- Postprocess
  |   |-- Remove posteriors padding (if needed)
  |   |-- Set posteriors outside the biggest connected component to zero (optional, '--keep_biggest_component')
  |   |-- Set posteriors outside the largest connected component of each topological class to zero (optional, '--use_topology_classes')
  |   |-- Get hard segmentation
  |   |-- Combine segmentation and parcellation (optional, '--parc parc.pth')
  |-- Output segmentations and posteriors
  ```

  **Notes:** The parcellation model is converted from the SynthSeg+ Tensorflow model. \
             **Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets** \
             B. Billot, M. Colin, Y. Cheng, S.E. Arnold, S. Das, J.E. Iglesias \
             PNAS (2023) \
             [ [article](https://www.pnas.org/doi/full/10.1073/pnas.2216399120) | [arxiv](https://arxiv.org/abs/2203.01969) ]

---

## Extension Points

### Adding New Models

1. Create model class in `freeseg/models/`
   This can be a wrapper class providing the interface between Freeseg and the network implementation.
2. Implement required methods:
   **`__init__()`**: takes `model_arch_dict` as input. Required `model_arch_dict` keywords: `num_channels`, `nb_labels`, `nb_levels`, `ndims`.
   ```
       def __init__(self, model_arch_dict):
           self._model_arch_dict = {}
        
           # set network defaults
           self._setdefault_arch_dict()
           # update network parameters with user input
           self._update_arch_dict(model_arch_dict)
	   
	   ...

   ```
   **`_setdefault_arch_dict(self)`**: set network defaults in `self._model_arch_dict`
   **`_update_arch_dict(self, model_arch_dict)`**: update network parameters `self._model_arch_dict` with user input
   **`forward()`**: torch.nn.Module forward method
3. Implement required property:
   ```
       @property
       def arch_dict(self):
           return self._model_arch_dict   
   ```

### Adding New Augmentations

1. Create augmentation wrapper class in `freeseg/augmentation/`
2. (optional) Inherit from base augmentation wrapper class freeseg.augmentation.augmentbase.AugmentBase
3. Implement individual augmentation class (inherit from `torch.nn.Module`)
4. Add to valid augmentations list

### Adding New Metrics

1. Create metric class in `freeseg/metrics.py`
2. Inherit from `torch.nn.Module`
3. Implement `__init__` and `forward` methods

### Adding New Datasets

1. Create dataset class in `freeseg/datasets/`
2. Inherit from `torch.utils.data.Dataset`
3. Implement `__getitem__` and `__len__`

---

## Future Enhancements


---

## References

- **U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)**
- **PyTorch Documentation**: https://pytorch.org/docs/
- **TensorBoard**: https://www.tensorflow.org/tensorboard
- **FreeSurfer**: https://freesurfer.net/
- **SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining** \
B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias \
Medical Image Analysis (2023) \
[ [article](https://www.sciencedirect.com/science/article/pii/S1361841523000506) | [arxiv](https://arxiv.org/abs/2107.09559) ]
- **Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets** \
B. Billot, M. Colin, Y. Cheng, S.E. Arnold, S. Das, J.E. Iglesias \
PNAS (2023) \
[ [article](https://www.pnas.org/doi/full/10.1073/pnas.2216399120) | [arxiv](https://arxiv.org/abs/2203.01969) ]
- **Voxynth**: https://github.com/dalcalab/voxynth/
