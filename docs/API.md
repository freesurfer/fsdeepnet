# FreeSeg API Documentation (WIP)

This document provides API documentation for the FreeSeg package, a PyTorch-based deep learning pipeline.

## Table of Contents

- [Core Modules](#core-modules)
- [Training](#training)
- [Prediction](#prediction)
- [Evaluation](#evaluation)
- [Models](#models)
- [Datasets](#datasets)
- [Augmentation](#augmentation)
- [Metrics](#metrics)
- [Configuration](#configuration)
- [Utilities](#utilities)

---

## Core Modules

### `freeseg.config.Config`

Configuration management class for handling YAML configuration files and command-line arguments.

#### Methods

##### `Config.process(args, logger=None, require_train_outfolder=True, require_dataset_list=True, test_augment=False)`

Processes configuration from YAML file and command-line arguments.

**Parameters:**
- `args`: Command-line arguments object
- `logger`: Logger instance (optional)
- `require_train_outfolder`: Whether training output folder is required
- `require_dataset_list`: Whether dataset list file is required
- `test_augment`: Whether testing augmentation mode

**Returns:**
- `dict`: Processed configuration dictionary

**Example:**
```python
from freeseg.config import Config
config = Config.process(args, logger=mainlogger)
```

##### `Config.load(config_file)`

Loads a YAML configuration file.

**Parameters:**
- `config_file` (str): Path to YAML configuration file

**Returns:**
- `dict`: Configuration dictionary

##### `Config.update(config, args)`

Updates configuration dictionary with command-line arguments.

**Parameters:**
- `config` (dict): Configuration dictionary
- `args`: Command-line arguments object

**Returns:**
- `dict`: Updated configuration dictionary

##### `Config.save(config, cwd=None, cmd=None, saveas=None, indent=0, sort_keys=False, debug=False)`

Saves configuration to a YAML file.

**Parameters:**
- `config` (dict): Configuration dictionary
- `cwd` (str, optional): Current working directory
- `cmd` (str, optional): Command string
- `saveas` (str, optional): Output file path
- `indent` (int): Indentation level
- `sort_keys` (bool): Whether to sort keys
- `debug` (bool): Debug mode

---

## Training

### `freeseg.training.Training`

Main training class for model training and validation.

#### Constructor

```python
Training(
    train_output_folder,
    train_loader,
    model,
    model_arch_dict=None,
    train_dataset_dict=None,
    ctab=None,
    model_checkpoint=None,
    validation_loader=None,
    best_model_metric="dice",
    write_tensorboard_summary=False,
    device=None,
    gpu_index=None,
    preprocessing_device=None,
    report_moving_avg=False,
    debug=False
)
```

**Parameters:**
- `train_output_folder` (str): Directory to save training outputs
- `train_loader` (DataLoader): Training data loader
- `model` (nn.Module): PyTorch model to train
- `model_arch_dict` (dict, optional): Model architecture dictionary
- `train_dataset_dict` (dict, optional): Training dataset configuration
- `ctab` (str, optional): Color table file path
- `model_checkpoint` (str, optional): Path to checkpoint for resuming training
- `validation_loader` (DataLoader, optional): Validation data loader
- `best_model_metric` (str): Metric for selecting best model ("loss" or "dice")
- `write_tensorboard_summary` (bool): Whether to write TensorBoard summaries
- `device` (torch.device, optional): Training device
- `gpu_index` (int, optional): GPU index
- `preprocessing_device` (torch.device, optional): Preprocessing device
- `report_moving_avg` (bool): Whether to report moving averages
- `debug` (bool): Debug mode

#### Methods

##### `Training.setup(config, preload_dataset=False)`

Static method to set up training components.

**Parameters:**
- `config` (dict): Configuration dictionary
- `preload_dataset` (bool): Whether to preload dataset

**Returns:**
- Tuple of (config, train_loader, validation_loader, model, optimizer_cls, model_arch_dict)

##### `train_model(config, train_loader, model, optimizer_cls, validation_loader=None)`

Runs the training loop.

**Parameters:**
- `config` (dict): Configuration dictionary
- `train_loader` (DataLoader): Training data loader
- `model` (nn.Module): Model to train
- `optimizer_cls`: Optimizer class
- `validation_loader` (DataLoader, optional): Validation data loader

---

## Prediction

### `freeseg.prediction.Prediction`

Class for running inference and predictions on new images.

#### Constructor

```python
Prediction(
    device=None,
    ctab=None,
    topology_classes=None,
    debug=False,
    debug_feat=False,
    gc=False
)
```

**Parameters:**
- `device` (torch.device, optional): Device for inference
- `ctab` (str, optional): Color table file path
- `topology_classes` (array, optional): Topology classes array
- `debug` (bool): Debug mode
- `debug_feat` (bool): Debug features mode
- `gc` (bool): Garbage collection flag

#### Methods

##### `build_model(segmentation_checkpoint, parcellation_checkpoint=None, qc_checkpoint=None, flip=False, smooth_posteriors=False, smooth_sigma=0.5)`

Builds the inference model from checkpoints.

**Parameters:**
- `segmentation_checkpoint` (str): Path to segmentation model checkpoint
- `parcellation_checkpoint` (str, optional): Path to parcellation model checkpoint
- `qc_checkpoint` (str, optional): Path to QC model checkpoint
- `flip` (bool): Whether to use left-right flipping
- `smooth_posteriors` (bool): Whether to smooth posterior probabilities
- `smooth_sigma` (float): Smoothing sigma value

##### `predict(image_path, output_path, label_path=None, prior_path=None, gt_path=None, crop_size=None, write_posteriors=False)`

Runs prediction on an image.

**Parameters:**
- `image_path` (str): Path to input image
- `output_path` (str): Path to save output segmentation
- `label_path` (str, optional): Path to label file
- `prior_path` (str, optional): Path to prior file
- `gt_path` (str, optional): Path to ground truth for evaluation
- `crop_size` (list, optional): Crop size for inference
- `write_posteriors` (bool): Whether to write posterior probabilities

---

## Evaluation

### `freeseg.evaluation.Evaluation`

Class for evaluating segmentation results using Dice scores.

#### Constructor

```python
Evaluation(labels_segmentation)
```

**Parameters:**
- `labels_segmentation` (array): Array of segmentation labels

#### Methods

##### `evaluate(gt_folder, eval_folder, evaluation_labels=None, path_dice=None)`

Evaluates segmentations in directories.

**Parameters:**
- `gt_folder` (str): Ground truth directory
- `eval_folder` (str): Predicted segmentation directory
- `evaluation_labels` (array, optional): Subset of labels to evaluate
- `path_dice` (str, optional): Path to save Dice scores as numpy array

**Returns:**
- `numpy.ndarray`: Dice scores array (rows=structures, columns=subjects)

##### `evaluate_oneseg(gt_path, seg_path, evaluation_labels=None)`

Evaluates a single segmentation.

**Parameters:**
- `gt_path` (str): Ground truth file path
- `seg_path` (str): Predicted segmentation file path
- `evaluation_labels` (array, optional): Labels to evaluate

**Returns:**
- `dict`: Dictionary mapping labels to Dice scores

---

## Models

### `freeseg.models.unet.UNet`

U-Net architecture for medical image segmentation.

#### Constructor

```python
UNet(model_arch_dict)
```

**Parameters:**
- `model_arch_dict` (dict): Model architecture configuration dictionary

**Required keys in `model_arch_dict`:**
- `num_channels` (int): Number of input channels
- `ndims` (int): Number of dimensions (2 or 3)
- `nb_features` (int): Number of base features
- `nb_levels` (int): Number of U-Net levels
- `nb_labels` (int): Number of output labels
- `feat_mult` (float): Feature multiplier
- `conv_size` (int): Convolution kernel size
- `pool_size` (int): Pooling size
- `nb_conv_per_level` (int): Number of convolutions per level
- `use_residuals` (bool): Whether to use residual connections
- `activation` (str): Activation function ("elu" or "relu")
- `weight_init` (str): Weight initialization method
- `norm` (str): Normalization type ("batch" or "instance")
- `final_pred_activation` (str): Final prediction activation ("softmax" or "sigmoid")

#### Methods

##### `forward(x)`

Forward pass through the network.

**Parameters:**
- `x` (torch.Tensor): Input tensor [N, C, H, W(, D)]

**Returns:**
- `torch.Tensor`: Output predictions [N, nb_labels, H, W(, D)]

### `freeseg.models.unet.ConvBlock`

Convolutional block used in U-Net.

#### Constructor

```python
ConvBlock(
    in_channels,
    out_channels,
    ndims=3,
    conv_size=3,
    nb_conv_per_level=1,
    use_residuals=False,
    activation="elu",
    weight_init="xavier_uniform"
)
```

**Parameters:**
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels
- `ndims` (int): Number of dimensions (2 or 3)
- `conv_size` (int): Convolution kernel size
- `nb_conv_per_level` (int): Number of convolutions
- `use_residuals` (bool): Whether to use residual connections
- `activation` (str): Activation function
- `weight_init` (str): Weight initialization method

---

## Datasets

### `freeseg.datasets.segmentationdataset.SegmentationDataset`

PyTorch Dataset class for segmentation tasks.

#### Constructor

```python
SegmentationDataset(
    dset_profile,
    augment_obj,
    dataset_dict,
    device=None,
    keep_trainset_in_memory=False,
    preload=False,
    augdir=None
)
```

**Parameters:**
- `dset_profile` (dict): Dataset profile configuration
- `augment_obj`: Augmentation object
- `dataset_dict` (list): List of dataset entries with image/label/prior filepaths
- `device` (torch.device, optional): Device for data loading
- `keep_trainset_in_memory` (bool): Whether to keep dataset in memory
- `preload` (bool): Whether to preload all data
- `augdir` (str, optional): Directory for augmented data

#### Methods

##### `__getitem__(idx)`

Returns a data sample.

**Parameters:**
- `idx` (int): Sample index

**Returns:**
- `dict`: Dictionary containing 'image', 'label', and optionally 'prior' tensors

##### `__len__()`

Returns dataset size.

**Returns:**
- `int`: Number of samples in dataset

---

## Augmentation

### `freeseg.augmentation.augmentbase.AugmentBase`

Base augmentation class providing various data augmentation transforms.

#### Constructor

```python
AugmentBase(
    hp,
    transforms,
    crop_size,
    num_channels=1,
    left_right_corresponding=None,
    bbox_labels=None,
    generation_labels=None,
    generation_classes=None,
    segmentation_labels=None,
    target_res=None,
    output_dir=None,
    device=None,
    sampling_hp=True,
    verbose=False
)
```

**Parameters:**
- `hp` (dict): Hyperparameters dictionary
- `transforms` (list): List of augmentation transforms to apply
- `crop_size` (list): Crop size [W, H(, D)]
- `num_channels` (int): Number of image channels
- `left_right_corresponding` (list, optional): Left-right label correspondences
- `bbox_labels` (list, optional): Labels for bounding box computation
- `generation_labels` (array, optional): Labels for generation
- `generation_classes` (array, optional): Classes for generation
- `segmentation_labels` (array, optional): Segmentation labels
- `target_res` (array, optional): Target resolution
- `output_dir` (str, optional): Output directory for augmented data
- `device` (torch.device, optional): Device for augmentation
- `sampling_hp` (bool): Whether to sample hyperparameters
- `verbose` (bool): Verbose mode

#### Available Augmentations

- `spatialdeformation`: Affine and non-linear spatial deformations
- `centercrop`: Center cropping with random offsets
- `randomcrop`: Random cropping
- `flip`: Left-right flipping
- `biasfieldcorruption`: Bias field corruption
- `intensityaugmentation`: Intensity augmentation (noise, gamma scaling)
- `mimicresolution`: Resolution mimicking
- `sampleconditionalgmm`: Conditional GMM sampling
- `remaplabels`: Label remapping

### `freeseg.augmentation.augmentvoxynth.AugmentVoxynth`

Extended augmentation class using Voxynth library (https://github.com/dalcalab/voxynth/).

---

## Metrics

### `freeseg.metrics.Dice`

Dice score and loss calculation.

#### Constructor

```python
Dice(
    num_classes,
    dice_type="soft",
    smooth=1e-6,
    return_loss=True,
    dice_squared_form=False
)
```

**Parameters:**
- `num_classes` (int): Number of segmentation classes
- `dice_type` (str): Type of Dice calculation ("soft" or "hard")
- `smooth` (float): Smoothing factor
- `return_loss` (bool): If True, returns Dice loss (1 - Dice)
- `dice_squared_form` (bool): Whether to use squared form

#### Methods

##### `forward(outputs, targets)`

Calculates Dice score or loss.

**Parameters:**
- `outputs` (torch.Tensor): Predicted probabilities [N, C, H, W(, D)]
- `targets` (torch.Tensor): Ground truth one-hot encoded [N, C, H, W(, D)]

**Returns:**
- `torch.Tensor`: Dice score or loss

### `freeseg.metrics.DiceLoss`

Dice loss wrapper (returns `1 - Dice`).

### `freeseg.metrics.DiceScore`

Dice score wrapper (returns `Dice`).

### `freeseg.metrics.WeightedL2Loss`

Weighted L2 loss for pre-training.

---

## Configuration

### Configuration File Structure

The configuration file (`config.yaml`) contains the following sections:

#### `dataset`
- `class_name`: Dataset class name
- `segmentation_labels`: List of segmentation labels
- `segmentation_names`: Label names (optional)
- `expected_num_channels`: Expected number of input channels
- `dataset_list_file`: Path to dataset list YAML file

#### `dataloader`
- `num_workers`: Number of data loading workers
- `pin_memory`: Whether to pin memory
- `persistent_workers`: Whether to use persistent workers
- `prefetch_factor`: Prefetch factor

#### `model`
- `name`: Model class name
- `nb_levels`: Number of U-Net levels
- `nb_features`: Base number of features
- `feat_mult`: Feature multiplier
- `nb_conv_per_level`: Number of convolutions per level
- `ndims`: Number of dimensions (2 or 3)
- `conv_size`: Convolution kernel size
- `pool_size`: Pooling size
- `use_residuals`: Whether to use residual connections
- `activation`: Activation function
- `weight_init`: Weight initialization method

#### `preprocessing`
- `augmentation_wrapper`: Augmentation class name
- `crop_size`: Crop size [W, H(, D)]
- `augmentations`: List of augmentation specifications

#### `training`
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `wl2_epochs`: Number of weighted L2 pre-training epochs
- `dice_epochs`: Number of Dice loss training epochs
- `steps_per_epoch`: Steps per epoch
- `optimizer`: Optimizer class name
- `model_metrics`: Loss/metric class name
- `write_tensorboard_summary`: Whether to write TensorBoard logs
- `perform_evaluation`: Whether to perform evaluation during training

---

## Utilities

### `freeseg.utils.utility`

Utility functions for data loading, logging, and other operations.

#### Functions

##### `load_framedimage(filepath, orientation="RAS", device=None, ndims=3)`

Loads a medical image file.

**Parameters:**
- `filepath` (str): Path to image file
- `orientation` (str): Target orientation
- `device` (torch.device, optional): Device to load data on
- `ndims` (int): Number of dimensions

**Returns:**
- Tuple of (FramedImage, tensor, geometry)

##### `config_logger(logfile=None, level=logging.INFO)`

Configures the root logger.

**Parameters:**
- `logfile` (str, optional): Log file path
- `level`: Logging level

### `freeseg.checkpoint.Checkpoint`

Checkpoint management class.

#### Methods

##### `load(checkpoint, model=None, optimizer=None, device=None)`

Loads a checkpoint file.

**Parameters:**
- `checkpoint` (str): Path to checkpoint file
- `model` (nn.Module, optional): Model to load weights into
- `optimizer` (optim.Optimizer, optional): Optimizer to load state into
- `device` (torch.device, optional): Device to load on

##### `save(checkpoint, dict)`

Saves a checkpoint.

**Parameters:**
- `checkpoint` (str): Path to save checkpoint
- `dict` (dict): Dictionary of values to save

---

## Scripts

### Command-Line Scripts

#### `freeseg_train.py`

Training script.

**Usage:**
```bash
fspython scripts/freeseg_train.py --config <config.yaml> [options]
```

#### `freeseg_predict.py`

Prediction script.

**Usage:**
```bash
fspython scripts/freeseg_predict.py --i <image> --o <output> --checkpoint <checkpoint> [options]
```

#### `freeseg_evaluate.py`

Evaluation script.

**Usage:**
```bash
fspython scripts/freeseg_evaluate.py --gt <ground_truth> --seg <segmentation> [options]
```

---

## Examples

### Training Example

```python
from freeseg.training import Training
from freeseg.config import Config
import argparse

args = argparse.Namespace()
args.config = "configs/config.yaml"
args.train_output_folder = "output/training"
args.dataset_list_file = "data/dataset_list.yaml"

config = Config.process(args)
config, train_loader, val_loader, model, optimizer_cls, _ = Training.setup(config)

trainer = Training(
    train_output_folder=config["output_folder"],
    train_loader=train_loader,
    model=model,
    model_arch_dict=config["model"],
    train_dataset_dict=config["dataset"],
    validation_loader=val_loader
)

trainer.train_model(config, train_loader, model, optimizer_cls, val_loader)
```

### Prediction Example

```python
from freeseg.prediction import Prediction

predictor = Prediction(device=torch.device("cuda"))
predictor.build_model(segmentation_checkpoint="checkpoints/best_model.pt")
predictor.predict(
    image_path="data/test_image.mgz",
    output_path="output/segmentation.mgz"
)
```

### Evaluation Example

```python
from freeseg.evaluation import Evaluation
import numpy as np

labels = np.array([0, 2, 3, 4, 17, 41])
evaluator = Evaluation(labels)

dice_scores = evaluator.evaluate(
    gt_folder="data/ground_truth",
    eval_folder="output/segmentations",
    path_dice="output/dice_scores.npy"
)
```

