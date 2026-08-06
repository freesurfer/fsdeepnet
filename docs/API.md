# Fsdeepnet API Documentation (WIP)

This document provides API documentation for the Fsdeepnet package, a PyTorch-based deep learning pipeline.

## Table of Contents

- [Core Modules](#core-modules)
- [Training](#training)
- [Prediction](#prediction)
- [Evaluation](#evaluation)
- [Models](#models)
- [Datasets](#datasets)
- [Augmentation](#augmentation)
- [Metrics](#metrics)
- [Filter](#filter)
- [Configuration](#configuration)
- [Utilities](#utilities)

---

## Core Modules

### `fsdeepnet.config.Config`

Configuration management class for handling YAML configuration files and command-line arguments. Provides utilities for loading, updating, saving, and processing configuration dictionaries used throughout the Fsdeepnet pipeline.

#### Methods

##### `Config.process(args, logger=None, require_train_outfolder=True, require_dataset_list=True, test_augment=False)`

Main method for processing configuration from YAML file and command-line arguments. This is the primary entry point for configuration management.

**Parameters:**
- `args` (argparse.Namespace or object): Command-line arguments object with attributes like `config`, `checkpoint`, `train_output_folder`, etc.
- `logger` (logging.Logger, optional): Logger instance for output. If `None`, uses the root logger
- `require_train_outfolder` (bool): Whether training output folder is required (default: `True`)
- `require_dataset_list` (bool): Whether dataset list file is required (default: `True`)
- `test_augment` (bool): Whether in augmentation testing mode (default: `False`)

**Returns:**
- `dict`: Processed configuration dictionary with all settings merged and validated

**Raises:**
- `AssertionError`: If required files/folders are missing or invalid
- `SystemExit`: If checkpoint file doesn't exist

**Behavior:**
1. Loads YAML configuration file specified in `args.config`
2. Updates configuration with command-line arguments via `Config.update()`
3. Validates checkpoint file if provided
4. Validates output folder and dataset list file requirements
5. Validates `crop_size` is divisible by `2^nb_levels` and matches `ndims`
6. Sets up logging and output directories
7. Saves copies of input config and dataset list to output folder
8. Configures training and preprocessing devices (CPU/GPU)
9. Updates configuration with runtime information (command, timestamp, devices, etc.)

**Configuration Keys Added/Updated:**
- `cmd`: Command string
- `cwd`: Current working directory
- `now`: Current datetime
- `device`: Training device (torch.device)
- `gpu_index`: GPU index (int or None)
- `preprocessing_device`: Preprocessing device (torch.device)
- `checkpoint`: Checkpoint path (str or None)
- `ctab`: Color table path (str or None)
- `keep_trainset_in_memory`: Whether to keep dataset in memory (bool)
- `logfile`: Log file path (str or None)
- `output_folder`: Output folder path (str or None)
- `train_augmentations`: Training augmentations list (initially None)
- `vmp`: Virtual memory profiling flag (bool)
- `debug`: Debug mode flag (bool)
- `verbose`: Verbose mode flag (bool)
- `train_cohort`: Training cohort name (str or None)
- `validation_cohort`: Validation cohort name (str or None)
- `config_saveas`: Saved config file path (str or None)
- `dataset_list_saveas`: Saved dataset list file path (str or None)

**Example:**
```python
from fsdeepnet.config import Config
import argparse

args = argparse.Namespace()
args.config = "configs/config.yaml"
args.train_output_folder = "output/training"
args.dataset_list_file = "data/dataset_list.yaml"

config = Config.process(args, logger=mainlogger)
```

##### `Config.load(config_file)`

Loads a YAML configuration file.

**Parameters:**
- `config_file` (str): Path to YAML configuration file

**Returns:**
- `dict`: Configuration dictionary loaded from YAML file

**Raises:**
- `AssertionError`: If the configuration file doesn't exist

**Example:**
```python
config = Config.load("configs/config.yaml")
```

##### `Config.update(config, args)`

Updates configuration dictionary with command-line arguments. Handles backward compatibility and sets default values for DataLoader parameters.

**Parameters:**
- `config` (dict): Configuration dictionary (typically loaded from YAML)
- `args` (argparse.Namespace or object): Command-line arguments object

**Returns:**
- `dict`: Updated configuration dictionary

**Behavior:**
- Creates `dataloader` section if missing (backward compatibility)
- Overwrites config values with command-line arguments if provided
- Sets default DataLoader parameters:
  - `num_workers`: Default 0
  - `pin_memory`: Default False
  - `persistent_workers`: Default False (set to False if `num_workers=0`)
  - `prefetch_factor`: Default 2 (set to None if `num_workers=0`)

**Command-Line Arguments Handled:**
- Model: `model_name`, `weight_init`, `nb_levels`, `nb_features`, `feat_mult`, `nb_conv_per_level`, `conv_size`, `pool_size`, `use_residuals`
- Training: `wl2_epochs`, `dice_epochs`, `learning_rate`, `batch_size`, `train_output_folder`, `report_moving_avg`, `deterministic`, `write_tensorboard_summary`, `perform_evaluation`, `best_model_metric`
- Dataset: `dataset_list_file`, `res_diff_thresh`
- Preprocessing: `crop_size`, `augmentation_dir`, `verbose`
- DataLoader: `num_workers`, `prefetch_factor`, `pin_memory`, `persistent_workers`

**Note:** Only updates config if the argument exists in `args` and is not `None` (or `True` for boolean flags). This allows the method to be shared across different scripts with different argument sets.

##### `Config.save(config, cwd=None, cmd=None, saveas=None, indent=0, sort_keys=False, debug=False)`

Saves configuration dictionary to a YAML file with optional metadata header.

**Parameters:**
- `config` (dict): Configuration dictionary to save
- `cwd` (str, optional): Current working directory to include in header
- `cmd` (str, optional): Command string to include in header
- `saveas` (str, optional): Output file path. If `None`, writes to stdout
- `indent` (int): Indentation level for YAML output (default: `0`)
- `sort_keys` (bool): Whether to sort dictionary keys (default: `False`, not yet implemented)
- `debug` (bool): Enable debug output during dumping (default: `False`)

**Behavior:**
- Writes metadata header with timestamp, CWD, and command
- Uses custom `Config.dump()` method for YAML formatting
- Creates output file if `saveas` is provided, otherwise writes to stdout

**Example:**
```python
Config.save(config, cwd=os.getcwd(), cmd=" ".join(sys.argv), 
            saveas="output/config.yaml")
```

##### `Config.print(cfg, logger=None)`

Prints configuration summary to logger.

**Parameters:**
- `cfg` (dict): Configuration dictionary to print
- `logger` (logging.Logger, optional): Logger instance. If `None`, uses root logger

**Output:**
Prints the following information:
- Training device and GPU index
- Checkpoint information (if resuming)
- Training epochs and learning rates (wl2 and dice stages)
- Steps per epoch, batch size, report_moving_avg
- Dataset settings (keep_trainset_in_memory, deterministic)
- Evaluation settings (if enabled)
- Preprocessing device and settings
- Augmentation wrapper and augmentations
- Crop size, sampling_hp
- DataLoader settings (num_workers, persistent_workers, pin_memory, prefetch_factor)
- Color table, output folder, config/dataset list save paths, log file

##### `Config.list2dict(in_list)`

Converts a list to a dictionary. Used for processing augmentation specifications.

**Parameters:**
- `in_list` (list): Input list containing strings or dictionaries

**Returns:**
- `dict`: Dictionary where:
  - String items become keys with empty dict values: `{item: {}}`
  - Dictionary items are merged into the output dictionary

**Example:**
```python
aug_list = ["spatialdeformation", {"intensityaugmentation": {"sigma": 0.1}}]
aug_dict = Config.list2dict(aug_list)
# Returns: {"spatialdeformation": {}, "intensityaugmentation": {"sigma": 0.1}}
```

##### `Config.get_augmentations(aug_list)`

Extracts augmentation names from an augmentation list specification.

**Parameters:**
- `aug_list` (list): List of augmentation specifications (strings or dicts)

**Returns:**
- `dict_keys`: Keys from the dictionary created by `list2dict()`, representing augmentation names

**Example:**
```python
aug_list = ["spatialdeformation", "intensityaugmentation"]
aug_names = Config.get_augmentations(aug_list)
# Returns: dict_keys(['spatialdeformation', 'intensityaugmentation'])
```

##### `Config.dump(data, fp=sys.stdout, indent=0, sort_keys=False, parent=None, listidx=0, debug=False)`

Recursively dumps data structure to file in YAML-like format. Internal method used by `Config.save()`.

**Parameters:**
- `data` (dict, list, tuple, or other): Data structure to dump
- `fp` (file-like object): File object to write to (default: `sys.stdout`)
- `indent` (int): Current indentation level (default: `0`)
- `sort_keys` (bool): Whether to sort keys (default: `False`, not yet implemented)
- `parent` (dict, list, or None): Parent data structure type for formatting (default: `None`)
- `listidx` (int): Current list index (default: `0`)
- `debug` (bool): Enable debug output (default: `False`)

**Behavior:**
- Formats dictionaries with proper indentation
- Formats lists with bracket notation `[item1, item2, ...]`
- Handles nested structures recursively
- Special formatting for augmentation specifications (lists of dicts)

**Note:** This is a low-level method typically called by `Config.save()`. Use `Config.save()` for saving configurations.

##### `Config.load_dataset_list(dataset_list_file)`

Loads a dataset list YAML file.

**Parameters:**
- `dataset_list_file` (str): Path to dataset list YAML file

**Returns:**
- `dict`: Dictionary containing dataset cohorts (train, validation, test) with their entries

**Example:**
```python
dataset_dict = Config.load_dataset_list("data/dataset_list.yaml")
# Returns: {"train": [...], "validation": [...], "test": [...]}
```

##### `Config.retrieve_dataset_cohorts(dataset_dict, cohorts)`

Retrieves and combines dataset entries from specified cohorts.

**Parameters:**
- `dataset_dict` (dict): Dataset dictionary loaded from YAML file
- `cohorts` (list): List of cohort names to retrieve (e.g., `["train", "validation"]`)

**Returns:**
- `list`: Combined list of dataset entries from all specified cohorts

**Example:**
```python
dataset_dict = {"train": [entry1, entry2], "validation": [entry3]}
cohorts = ["train", "validation"]
dataset = Config.retrieve_dataset_cohorts(dataset_dict, cohorts)
# Returns: [entry1, entry2, entry3]
```

---

## Training

### `fsdeepnet.training.Training`

Main training class for model training and validation. Supports two-stage training with weighted L2 pre-training followed by Dice loss training.

#### Class Attributes

- `stage_order` (dict): Dictionary mapping metric types to stage order: `{'wl2': 0, 'dice': 1}`

#### Constructor

```python
Training(
    train_output_folder,
    train_loader,
    model,
    accuracy_fn=None,
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
- `train_output_folder` (str): Path to directory where models will be saved during training
- `train_loader` (torch.utils.data.DataLoader): DataLoader to create the training data generator
- `model` (nn.Module): PyTorch model to train
- `accuracy_fn` (callable, optional): Accuracy function for computing Dice scores (e.g., `DiceScore`)
- `model_arch_dict` (dict, optional): Model architecture dictionary containing model configuration
- `train_dataset_dict` (dict, optional): Training dataset configuration dictionary
- `ctab` (str, optional): Path to ASCII color table file for label visualization
- `model_checkpoint` (str, optional): Path to an already saved model to load before starting training
- `validation_loader` (torch.utils.data.DataLoader, optional): Validation DataLoader for evaluation during training
- `best_model_metric` (str): Metric for selecting best model - either `"loss"` or `"dice"` (default: `"dice"`)
- `write_tensorboard_summary` (bool): Whether to write TensorBoard summaries (default: `False`)
- `device` (torch.device, optional): Device for training. If `None`, automatically selects CUDA if available, else CPU
- `gpu_index` (int, optional): GPU index to use. If `None` and CUDA is available, uses current device
- `preprocessing_device` (torch.device, optional): Device for data preprocessing. If `None`, uses same as `device`
- `report_moving_avg` (bool): Whether to report moving average of loss/dice instead of per-step values (default: `False`)
- `debug` (bool): Enable debug mode to save intermediate volumes and predictions (default: `False`)

**Raises:**
- `AssertionError`: If `best_model_metric == "dice"` and `validation_loader` is provided but `accuracy_fn` is `None`

#### Methods

##### `train_model(lr, epochs, steps_per_epoch, metric_type, optimizer_cls, loss_fn)`

Runs the training loop for the specified number of epochs.

**Parameters:**
- `lr` (float): Learning rate for the training
- `epochs` (int): Number of epochs for which the network is trained
- `steps_per_epoch` (int): Number of training steps per epoch. This determines the frequency at which models are saved
- `metric_type` (str): Training metric type - either `"wl2"` (weighted L2) or `"dice"` (Dice loss)
- `optimizer_cls` (class): Optimizer class (e.g., `torch.optim.Adam`, `torch.optim.SGD`)
- `loss_fn` (nn.Module): Loss function module (e.g., `WeightedL2Loss`, `DiceLoss`)

**Behavior:**
- If `model_checkpoint` is provided, resumes training from the checkpoint
- Saves checkpoints after each epoch to `{metric_type}_{epoch:03d}.pth`
- If validation is enabled, performs validation after each training epoch
- Saves best model based on `best_model_metric` (either lowest loss or highest Dice score)
- Saves training/validation metrics to numpy arrays and text files
- Writes TensorBoard summaries if `write_tensorboard_summary=True`
- Outputs debug volumes if `debug=True`

**Output Files:**
- Checkpoints: `{checkpoint_dir}/{metric_type}_{epoch:03d}.pth`
- Best models: `{best_model_dir}/best_{metric}_model_{metric_type}_{epoch:03d}.pth`
- Dice scores: `{dice_dir}/train_{metric_type}_{epoch:03d}.npy` and `validation_{metric_type}_{epoch:03d}.npy`
- Average metrics: `{checkpoint_dir}/train_validation_avg_{metric_type}_epoch{start}-{end}.dat`

##### `Training.setup(config, preload_dataset=False, create_train_dataset=True, create_loader=True, create_val_loader=True, create_model=True)`

Static method to set up training components including datasets, data loaders, model, and optimizer.

**Parameters:**
- `config` (dict): Configuration dictionary containing all training parameters
- `preload_dataset` (bool): Whether to preload the entire dataset into memory (default: `False`)
- `create_train_dataset` (bool): Whether to create training dataset (default: `True`)
- `create_loader` (bool): Whether to create data loaders (default: `True`)
- `create_val_loader` (bool): Whether to create validation loader (default: `True`)
- `create_model` (bool): Whether to create model (default: `True`)

**Returns:**
- `tuple`: Tuple containing:
  - `config` (dict): Updated configuration dictionary
  - `train_loader` (DataLoader or None): Training data loader
  - `validation_loader` (DataLoader or None): Validation data loader
  - `model` (nn.Module or None): PyTorch model instance
  - `optimizer_cls` (class or None): Optimizer class
  - `train_dataset` (Dataset or None): Training dataset object

**Configuration Keys Used:**
- `config["dataset"]`: Dataset configuration
- `config["preprocessing"]`: Preprocessing and augmentation settings
- `config["evaluation"]`: Evaluation/validation settings
- `config["training"]`: Training hyperparameters (batch_size, optimizer, etc.)
- `config["model"]`: Model architecture configuration
- `config["dataloader"]`: DataLoader settings (num_workers, pin_memory, etc.)

**Side Effects:**
- Updates `config` dictionary with processed dataset attributes
- Creates output directories if `config["output_folder"]` is specified
- Saves `reported_generation_labels.npy` if `preload_dataset=True`
- Sets deterministic training if `config["training"]["deterministic"]` is `True`

#### Private Methods

##### `_train_one_epoch(optimizer, loss_fn, epoch, steps_per_epoch, metric_type='dice')`

Trains the model for one epoch.

**Parameters:**
- `optimizer` (torch.optim.Optimizer): Optimizer instance
- `loss_fn` (nn.Module): Loss function module
- `epoch` (int): Current epoch number
- `steps_per_epoch` (int): Number of training steps per epoch
- `metric_type` (str): Metric type - `"wl2"` or `"dice"` (default: `"dice"`)

**Returns:**
- `tuple`: Tuple containing:
  - `train_loss` (float): Total training loss for the epoch
  - `train_dices` (numpy.ndarray or None): Array of Dice scores with shape `(batch_size, num_labels, steps_per_epoch)` if `accuracy_fn` is provided, else `None`

##### `_validate(optimizer, loss_fn, epoch, metric_type='dice')`

Validates the model on the validation set.

**Parameters:**
- `optimizer` (torch.optim.Optimizer): Optimizer instance (not used but kept for API consistency)
- `loss_fn` (nn.Module): Loss function module
- `epoch` (int): Current epoch number
- `metric_type` (str): Metric type - `"wl2"` or `"dice"` (default: `"dice"`)

**Returns:**
- `tuple`: Tuple containing:
  - `validation_loss` (float): Total validation loss
  - `validation_dices` (numpy.ndarray or None): Array of Dice scores with shape `(batch_size, num_labels, num_batches)` if `accuracy_fn` is provided, else `None`

**Note:** This method is called automatically during training if `validation_loader` is provided.

---

## Prediction

### `fsdeepnet.prediction.Prediction`

Class for running inference and predictions on new images. Supports segmentation, parcellation, and quality control (QC) models. Can process single images or batches of images from directories.

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
- `device` (torch.device, optional): Device for inference. If `None`, automatically selects CUDA if available, else CPU
- `ctab` (str, optional): Path to color table (LUT) file for label visualization. Overrides the color table saved in checkpoint
- `topology_classes` (numpy.ndarray or str, optional): Array of topology classes for each segmentation label, or path to `.npy` file. Overrides the topology classes saved in checkpoint
- `debug` (bool): Enable debug mode to save intermediate volumes and preprocessing steps (default: `False`)
- `debug_feat` (bool): Enable debug mode to save intermediate feature maps from model layers (default: `False`)
- `gc` (bool): Enable garbage collection after model forward passes to free memory (default: `False`)

#### Methods

##### `build_model(segmentation_checkpoint, parcellation_checkpoint=None, qc_checkpoint=None, flip=False, smooth_posteriors=False, smooth_sigma=0.5)`

Builds and assembles the inference model from checkpoint files. The inference model can include segmentation, optional parcellation, and optional QC models with optional left-right flipping and posterior smoothing.

**Parameters:**
- `segmentation_checkpoint` (str): Path to segmentation model checkpoint file (required)
- `parcellation_checkpoint` (str, optional): Path to parcellation model checkpoint file
- `qc_checkpoint` (str, optional): Path to quality control model checkpoint file
- `flip` (bool): Whether to use left-right flipped image prediction and average with original (default: `False`). Requires `left_right_corresponding` in checkpoint
- `smooth_posteriors` (bool): Whether to apply Gaussian smoothing to posterior probabilities (default: `False`)
- `smooth_sigma` (float): Standard deviation for Gaussian smoothing kernel (default: `0.5`)

**Raises:**
- `AssertionError`: If checkpoint file does not exist or required information is missing
- `AssertionError`: If `flip=True` but `left_right_corresponding` is not available in checkpoint

**Behavior:**
- Loads segmentation model and extracts model architecture, dataset configuration, and label mappings
- If `flip=True`, computes posterior channel indices for left-right flipped labels
- If parcellation model is provided, concatenates parcellation labels with segmentation labels
- Creates `InferenceModel` wrapper that handles all inference operations
- Sets model to evaluation mode

**Note:** Must be called before `predict()`.

##### `predict(path_images, out_segmentations, crop_size=None, target_res=None, resample_thresh=None, path_labels=None, path_priors=None, codenames=None, path_gt=None, addctab=True, write_posteriors=False, path_volumes=None, keepgeom=True, keep_biggest_component=False, topology_classes=None, segmentation_names=None, use_topology_classes=False)`

Runs prediction on one or more images. Supports single files, lists of files, or directories of images.

**Parameters:**
- `path_images` (str, list, or directory): Path to input image(s). Can be:
  - Single file path (e.g., `"image.mgz"`)
  - List of file paths
  - Directory path containing images (`.nii.gz`, `.nii`, or `.mgz`)
- `out_segmentations` (str or directory): Path to save output segmentation(s). Must be:
  - Single file path if `path_images` is a single file
  - Directory path if `path_images` is a list or directory
- `crop_size` (list, optional): Crop size `[W, H(, D)]` for inference. If `None`, uses full image size. Automatically adjusted to be divisible by `2^nb_levels`
- `target_res` (list, optional): Target resolution `[x, y(, z)]` in mm. If `None`, uses resolution from training checkpoint
- `resample_thresh` (float, optional): Threshold for resampling. If `None`, uses value from training checkpoint
- `path_labels` (str, list, or directory, optional): Path to label image(s) for centroid-based cropping. Must match structure of `path_images`
- `path_priors` (str, list, or directory, optional): Path to prior probability map(s). Must match structure of `path_images` and have shape `[num_labels, H, W(, D)]`
- `codenames` (list, optional): List of codenames for output files. If `None`, uses sequential numbers (0001, 0002, ...)
- `path_gt` (str, list, or directory, optional): Path to ground truth segmentation(s) for evaluation. If provided, calculates Dice scores
- `addctab` (bool): Whether to add color table to output segmentation files (default: `True`)
- `write_posteriors` (bool): Whether to save posterior probability maps (default: `False`)
- `path_volumes` (str, optional): Path to CSV file to save volume measurements for each label
- `keepgeom` (bool): Whether to resample output back to original image geometry (default: `True`)
- `keep_biggest_component` (bool): Whether to keep only the largest connected component in segmentation (default: `False`)
- `topology_classes` (numpy.ndarray, optional): Override topology classes for this prediction
- `segmentation_names` (numpy.ndarray, optional): Label names for volume output
- `use_topology_classes` (bool): Whether to keep largest connected component per topology class (default: `False`)

**Raises:**
- `AssertionError`: If `path_images` or `out_segmentations` is `None`
- `AssertionError`: If `use_topology_classes=True` but topology classes are not available
- `AssertionError`: If input/output path structures don't match (e.g., single file vs directory)
- `AssertionError`: If image and label shapes don't match when `path_labels` is provided
- `AssertionError`: If prior shape doesn't match expected shape `[num_labels, H, W(, D)]`

**Output Files:**
- Segmentation files: Saved to `out_segmentations` with suffix `.prediction.mgz` or `.prediction.nii`
- Posterior files (if `write_posteriors=True`): Saved to `posteriors/` subdirectory
- Volume CSV (if `path_volumes` provided): CSV file with volume measurements and `.stats` file with statistics
- Predictions list: `predictions.lst` file listing all processed images
- Dice scores (if `path_gt` provided): `dices.npy` file with Dice scores
- Debug files (if `debug=True`): Intermediate volumes saved to `debug/` subdirectory
- Feature maps (if `debug_feat=True`): Layer outputs saved to `debug/features/` subdirectory

**Processing Pipeline:**
1. **Preprocessing**: Load image, reorient to RAS, resample to target resolution, crop (if needed), normalize, pad
2. **Inference**: Run through inference model (segmentation ± parcellation ± QC)
3. **Postprocessing**: Remove padding, apply connected component filtering (if enabled), get hard segmentation, combine with parcellation (if applicable)
4. **Output**: Save segmentation, posteriors (if requested), volumes (if requested), and evaluation (if ground truth provided)

#### Private Methods

##### `load_segmentation_model(model_checkpoint)`

Loads the segmentation model from checkpoint.

**Parameters:**
- `model_checkpoint` (str): Path to segmentation model checkpoint

**Returns:**
- `nn.Module`: Loaded segmentation model in evaluation mode

**Side Effects:**
- Extracts and stores model architecture information (`nb_levels`, `ndims`)
- Extracts and stores dataset information (labels, names, mappings, topology classes)
- Sets `_target_res` and `_resample_thresh` from checkpoint

##### `load_parcellation_model(model_checkpoint)`

Loads the parcellation model from checkpoint.

**Parameters:**
- `model_checkpoint` (str): Path to parcellation model checkpoint

**Returns:**
- `nn.Module`: Loaded parcellation model in evaluation mode

**Side Effects:**
- Extracts and stores parcellation labels and names
- Creates parcellation label mappings

##### `load_qc_model(model_checkpoint)`

Loads the quality control model from checkpoint. Currently returns `None` (not implemented).

**Parameters:**
- `model_checkpoint` (str): Path to QC model checkpoint

**Returns:**
- `None`: QC model loading not yet implemented

##### `register_hook(module, name='')`

Registers forward hooks on model modules to save intermediate feature maps for debugging.

**Parameters:**
- `module` (nn.Module or list): Module(s) to register hooks on
- `name` (str): Name prefix for saved feature files

**Note:** Only used when `debug_feat=True`. Feature maps are saved to `debug/features/` directory.

##### `unregister_hook()`

Removes all registered forward hooks.

##### `preprocess(idx, path_images, path_labels, path_priors, codenames, label_lookup)`

Preprocesses a single image for inference.

**Parameters:**
- `idx` (int): Index of image to process
- `path_images` (list): List of image paths
- `path_labels` (list, optional): List of label paths
- `path_priors` (list, optional): List of prior paths
- `codenames` (list): List of codenames
- `label_lookup`: Label lookup object

**Returns:**
- `tuple`: Tuple containing:
  - `sfimage`: Original FramedImage object
  - `orig_orientation`: Original image orientation
  - `target_im_geom`: Target image geometry
  - `target_im_shape`: Target image shape
  - `image_tensor_preprocessed`: Preprocessed image tensor with batch dimension
  - `prior_tensor_preprocessed`: Preprocessed prior tensor with batch dimension (or `None`)
  - `crop_idx`: Crop indices (or `None`)
  - `pad_idx`: Padding indices
  - `label_lookup`: Label lookup object

##### `postprocess(posteriors_seg, target_im_res, target_im_shape, crop_idx, pad_idx, keep_biggest_component=False, use_topology_classes=False, path_volumes=None, posteriors_parc=None)`

Postprocesses model outputs to generate final segmentation.

**Parameters:**
- `posteriors_seg` (torch.Tensor): Segmentation posteriors `[B, C, H, W(, D)]`
- `target_im_res` (list): Target image resolution
- `target_im_shape` (tuple): Target image shape
- `crop_idx` (list or None): Crop indices to restore original image size
- `pad_idx` (list): Padding indices to remove
- `keep_biggest_component` (bool): Whether to keep only largest connected component
- `use_topology_classes` (bool): Whether to keep largest component per topology class
- `path_volumes` (str, optional): Path to save volume measurements
- `posteriors_parc` (torch.Tensor, optional): Parcellation posteriors

**Returns:**
- `tuple`: Tuple containing:
  - `segmentation` (torch.Tensor): Final segmentation `[B, H, W(, D)]`
  - `posteriors_seg` (torch.Tensor): Final posteriors `[B, C, H, W(, D)]`

##### `prepare_output_files(path_images, path_labels, path_priors, out_segmentations, codenames, write_posteriors, path_volumes)`

Prepares and validates output file paths based on input paths.

**Parameters:**
- `path_images` (str, list, or directory): Input image path(s)
- `path_labels` (str, list, or directory, optional): Label path(s)
- `path_priors` (str, list, or directory, optional): Prior path(s)
- `out_segmentations` (str or directory): Output segmentation path(s)
- `codenames` (list, optional): Codenames for outputs
- `write_posteriors` (bool): Whether to write posteriors
- `path_volumes` (str, optional): Path to volume CSV file

**Returns:**
- `tuple`: Tuple containing:
  - `path_images` (list): List of image paths
  - `path_labels` (list or None): List of label paths
  - `path_priors` (list or None): List of prior paths
  - `out_segmentations` (list): List of output segmentation paths
  - `codenames` (list): List of codenames
  - `out_posteriors` (list): List of output posterior paths (empty if `write_posteriors=False`)
  - `csv_subjects` (list): List of subject names for CSV output

##### `get_posteriors_flipped_indices()`

Computes posterior channel indices for left-right flipped labels. Called automatically by `build_model()` when `flip=True`.

**Side Effects:**
- Sets `_posterior_flipped_indices` attribute with channel mapping for flipped labels

---

### `fsdeepnet.prediction.InferenceModel`

Internal PyTorch module that wraps segmentation, parcellation, and QC models for inference.

#### Constructor

```python
InferenceModel(
    segmentation_model,
    label_mapping=None,
    posterior_flipped_indices=None,
    smooth_sigma=0.5,
    device=None,
    smooth_posteriors=False,
    parcellation_model=None,
    qc_model=None,
    gc=False
)
```

**Parameters:**
- `segmentation_model` (nn.Module): Segmentation model
- `label_mapping` (dict, optional): Mapping from label IDs to posterior channel indices
- `posterior_flipped_indices` (list, optional): Channel indices for left-right flipped labels
- `smooth_sigma` (float): Gaussian smoothing sigma (default: `0.5`)
- `device` (torch.device, optional): Device for inference
- `smooth_posteriors` (bool): Whether to smooth posteriors (default: `False`)
- `parcellation_model` (nn.Module, optional): Parcellation model
- `qc_model` (nn.Module, optional): Quality control model
- `gc` (bool): Enable garbage collection (default: `False`)

#### Methods

##### `forward(x, x1=None, predict_obj=None)`

Forward pass through the inference model.

**Parameters:**
- `x` (torch.Tensor): Input image tensor `[B, C, H, W(, D)]`
- `x1` (torch.Tensor, optional): Prior tensor `[B, C_prior, H, W(, D)]`
- `predict_obj` (Prediction, optional): Prediction object for debug output

**Returns:**
- `tuple`: Tuple containing:
  - `posteriors_seg` (torch.Tensor): Segmentation posteriors `[B, C, H, W(, D)]`
  - `posteriors_parc` (torch.Tensor or None): Parcellation posteriors (if parcellation model provided)
  - `qc_score` (torch.Tensor or None): QC score (if QC model provided)

**Behavior:**
- Runs segmentation model forward pass
- Optionally applies Gaussian smoothing to posteriors
- Optionally performs left-right flip and averages posteriors
- Optionally runs parcellation model on cortex regions
- Optionally runs QC model on segmentation posteriors

---

## Evaluation

### `fsdeepnet.evaluation.Evaluation`

Class for evaluating segmentation results using Dice scores. Computes Dice coefficients between ground truth and predicted segmentation label maps.

#### Constructor

```python
Evaluation(labels_segmentation)
```

**Parameters:**
- `labels_segmentation` (array-like): Array or list of segmentation labels to evaluate. Can contain duplicates; unique labels are extracted and sorted

**Behavior:**
- Extracts unique labels from `labels_segmentation` and stores them in sorted order
- Stores both the unique labels and their original indices

**Example:**
```python
labels = [0, 2, 3, 4, 17, 41, 42, 43, 53]
evaluator = Evaluation(labels)
```

#### Methods

##### `evaluate(gt_folder, eval_folder, evaluation_labels=None, path_dice=None)`

Performs Dice evaluation between ground truth and segmentation directories or files. Supports directories, lists of files, or single files.

**Parameters:**
- `gt_folder` (str, list, or directory): Path to ground truth label map(s). Can be:
  - Directory path: Evaluates all `.nii.gz`, `.nii`, or `.mgz` files in the directory
  - List of file paths: Evaluates files in the list
  - Single file path: Evaluates a single file
- `eval_folder` (str, list, or directory): Path to predicted segmentation(s). Must match the structure of `gt_folder`:
  - Directory path if `gt_folder` is a directory
  - List of file paths if `gt_folder` is a list
  - Single file path if `gt_folder` is a single file
- `evaluation_labels` (numpy.ndarray, optional): Subset of segmentation labels to evaluate. If `None`, uses all unique labels from constructor. Default: `None`
- `path_dice` (str, optional): Path to save Dice scores as numpy array. If `None`, saves to `{eval_folder}/dices.npy`. Default: `None`

**Returns:**
- `numpy.ndarray`: Dice scores array with shape `[n_labels, n_subjects]` where:
  - Rows correspond to structures (labels) in sorted order
  - Columns correspond to subjects (files evaluated)

**Raises:**
- `AssertionError`: If `gt_folder` or `eval_folder` is `None`
- `AssertionError`: If `gt_folder` and `eval_folder` are both directories but have different numbers of files
- `AssertionError`: If `gt_folder` or `eval_folder` is a file that doesn't exist

**Behavior:**
- Converts paths to absolute paths
- If directories are provided, finds all `.nii.gz`, `.nii`, and `.mgz` files and sorts them
- Evaluates each pair of ground truth and segmentation files using `evaluate_oneseg()`
- Saves results to multiple output files (see Output Files below)

**Output Files:**
- `{path_dice}.npy`: NumPy array with Dice scores `[n_labels, n_subjects]`
- `{path_dice}.dat`: Text file with Dice scores `[n_subjects, n_labels]` (transposed)
- `{path_dice}_evaluations.lst`: Text file listing evaluated segmentations with format `index:segmentation_path:avg_dice`
- `{path_dice}_avg_per_label.npy`: NumPy array with average Dice score per label `[n_labels]`
- `{path_dice}_avg_per_label.dat`: Text file with average Dice score per label

**Example:**
```python
evaluator = Evaluation([0, 2, 3, 4, 17, 41, 42, 43, 53])

# Evaluate directories
dice_scores = evaluator.evaluate(
    gt_folder="data/ground_truth",
    eval_folder="output/segmentations",
    path_dice="output/dices.npy"
)
# Returns: array with shape [9, n_subjects]

# Evaluate single files
dice_scores = evaluator.evaluate(
    gt_folder="data/gt.mgz",
    eval_folder="output/seg.mgz"
)

# Evaluate lists of files
dice_scores = evaluator.evaluate(
    gt_folder=["gt1.mgz", "gt2.mgz"],
    eval_folder=["seg1.mgz", "seg2.mgz"]
)
```

##### `evaluate_oneseg(path_gt, path_seg, evaluation_labels=None)`

Computes Dice score between a single ground truth label map and predicted segmentation.

**Parameters:**
- `path_gt` (str): Path to single ground truth label map file (`.nii.gz`, `.nii`, or `.mgz`)
- `path_seg` (str): Path to single predicted segmentation file (`.nii.gz`, `.nii`, or `.mgz`)
- `evaluation_labels` (numpy.ndarray, optional): Labels to evaluate. If `None`, uses all unique labels from constructor. Default: `None`

**Returns:**
- `numpy.ndarray`: Dice coefficients for each label with shape `[n_labels]`. Scores are in the same order as `evaluation_labels`

**Behavior:**
- Loads ground truth and segmentation volumes using `surfa` library
- Reorients both volumes to RAS orientation
- Computes Dice scores using `fast_dice()` function
- Logs evaluation progress

**Example:**
```python
evaluator = Evaluation([0, 2, 3, 4, 17, 41])
dice_scores = evaluator.evaluate_oneseg(
    path_gt="data/ground_truth.mgz",
    path_seg="output/segmentation.mgz"
)
# Returns: array([dice_0, dice_2, dice_3, dice_4, dice_17, dice_41])
```

---

### Functions

##### `fsdeepnet.evaluation.fast_dice(x, y, labels)`

Fast implementation of Dice scores using histogram-based computation.

**Parameters:**
- `x` (numpy.ndarray): Input label map (ground truth)
- `y` (numpy.ndarray): Input label map (prediction) of the same size as `x`
- `labels` (numpy.ndarray): Array of labels to evaluate on

**Returns:**
- `numpy.ndarray`: Dice scores in the same order as `labels`

**Raises:**
- `AssertionError`: If `x` and `y` have different shapes

**Behavior:**
- For multiple labels: Uses 2D histogram to compute intersections and unions efficiently
- For single label: Computes Dice directly using binary masks
- Uses smoothing factor `1e-5` to prevent division by zero

**Formula:**
For each label `l`:
```
Dice = (2 * intersection) / (sum(x==l) + sum(y==l) + 1e-5)
```

**Example:**
```python
gt = np.array([0, 0, 1, 1, 2, 2])
pred = np.array([0, 1, 1, 1, 2, 2])
labels = np.array([0, 1, 2])
dice_scores = fast_dice(gt, pred, labels)
# Returns: array([dice_0, dice_1, dice_2])
```

##### `fsdeepnet.evaluation.dice_coeffs(gt, pred, labels)`

Alternative implementation of Dice coefficient calculation using binary masks.

**Parameters:**
- `gt` (numpy.ndarray): Ground truth label map
- `pred` (numpy.ndarray): Predicted label map of the same size as `gt`
- `labels` (numpy.ndarray): Array of labels to evaluate

**Returns:**
- `numpy.ndarray`: Dice scores for each label with shape `[n_labels]`

**Raises:**
- `AssertionError`: If `gt` and `pred` have different shapes

**Behavior:**
- Converts ground truth and predictions to binary masks for each label
- Computes intersection and union for each label
- Uses smoothing factor `1e-5` to prevent division by zero

**Formula:**
For each label `l`:
```
intersection = sum((gt == l) * (pred == l))
union = sum(gt == l) + sum(pred == l)
Dice = (2 * intersection) / (union + 1e-5)
```

**Note:** This function is available but `fast_dice()` is used by default in `evaluate_oneseg()` for better performance.

---

## Models

### `fsdeepnet.models.unet.UNet`

U-Net architecture for medical image segmentation. Implements a fully convolutional encoder-decoder network with skip connections for dense prediction tasks.

#### Constructor

```python
UNet(model_arch_dict)
```

**Parameters:**
- `model_arch_dict` (dict): Model architecture configuration dictionary

**Required keys in `model_arch_dict`:**
- `num_channels` (int): Number of input channels. Can also be specified as `in_channels` to override dataset default
- `nb_labels` (int): Number of output labels/classes. Can also be specified as `out_channels` to override dataset default

**Optional keys in `model_arch_dict` (with defaults):**
- `ndims` (int): Number of dimensions (2 or 3). Default: `3`
- `nb_features` (int): Number of base features at the first level. Default: `24`
- `nb_levels` (int): Number of U-Net levels (encoder/decoder levels). Default: `3`
- `feat_mult` (float): Feature multiplier applied at each level. Features at level `i` = `nb_features * (feat_mult ** i)`. Default: `1`
- `conv_size` (int): Convolution kernel size. Default: `3`
- `pool_size` (int): Pooling/upsampling size. Default: `2`
- `nb_conv_per_level` (int): Number of convolutions per level. Default: `1`
- `use_residuals` (bool): Whether to use residual connections in ConvBlocks. Default: `False`
- `activation` (str): Activation function - `"elu"` or `"relu"`. Default: `"elu"`
- `weight_init` (str): Weight initialization method - `"xavier_uniform"` or `"zeros"`. Default: `"xavier_uniform"`
- `norm` (str or None): Normalization type - `"batch"`, `"instance"`, or `None`. Default: `None`
- `final_pred_activation` (str): Final prediction activation - `"softmax"`, `"sigmoid"`, or `"linear"`. Default: `"softmax"`
- `add_priors` (bool): Whether to add prior probability maps to output. Default: `False`
- `refine_conv` (bool): Whether to use refinement convolution in decoder. Default: `False`
- `track_running_stats` (bool): Whether to track running statistics in batch/instance normalization. Default: `False`
- `upsample_interpolation` (str): Upsampling interpolation mode - `"linear"` or `"nearest"`. Default: `"linear"`
- `skip_connect` (str): Skip connection source - `"norm"` (after normalization) or `"encoder"` (after ConvBlock). Default: `"norm"`. If `norm=None`, defaults to `"encoder"`

**Raises:**
- `AssertionError`: If `norm` is not `"batch"`, `"instance"`, or `None`
- `AssertionError`: If `weight_init` is not `"xavier_uniform"` or `"zeros"`
- `AssertionError`: If `upsample_interpolation` is not `"linear"` or `"nearest"`
- `AssertionError`: If `skip_connect` is not `"norm"` or `"encoder"`
- `AssertionError`: If `skip_connect="norm"` but `norm=None`
- `ValueError`: If `final_pred_activation` is not `"softmax"`, `"sigmoid"`, or `"linear"`

**Backward Compatibility:**
The constructor handles backward compatibility for older model configurations:
- `input_shape` → `num_channels` (extracts first element)
- `use_batchnorm` → `norm` (maps `True` to `"batch"`, `False` to `None`)
- `skip_connect_from` → `skip_connect` (renames, maps `"batchnorm"` to `"norm"`)
- `bn_track_running_stats` → `track_running_stats` (renames)

#### Methods

##### `forward(x, priors=None, **kwargs)`

Forward pass through the U-Net network.

**Parameters:**
- `x` (torch.Tensor): Input tensor with shape `[N, num_channels, H, W(, D)]` where:
  - `N`: Batch size
  - `num_channels`: Number of input channels
  - `H, W(, D)`: Spatial dimensions (2D or 3D)
- `priors` (torch.Tensor, optional): Prior probability maps with shape `[N, nb_labels, H, W(, D)]`. Only used if `add_priors=True`. Default: `None`
- `**kwargs`: Additional keyword arguments (ignored)

**Returns:**
- `list`: List containing two tensors:
  - `[0]` (torch.Tensor): Final predictions with shape `[N, nb_labels, H, W(, D)]` after final activation and prior addition (if enabled)
  - `[1]` (torch.Tensor): Penultimate layer output (before final activation) with shape `[N, nb_labels, H, W(, D)]`. Used for WeightedL2Loss training

**Architecture:**
1. **Encoder (Contracting path)**: 
   - ConvBlock → (Normalization) → MaxPooling
   - Features double at each level (if `feat_mult=2`)
   - Skip connections saved for decoder
2. **Bottleneck**: 
   - ConvBlock → (Normalization)
   - Deepest level with most features
3. **Decoder (Expansive path)**:
   - Upsample → (Refinement Conv) → Concatenate skip connection → ConvBlock → (Normalization)
   - Features halve at each level
   - Skip connections concatenated with upsampled features
4. **Classification**: 
   - 1x1 convolution to `nb_labels` channels
5. **Final activation**: 
   - Softmax, Sigmoid, or Linear (based on `final_pred_activation`)
6. **Prior addition** (if enabled): 
   - Adds priors to softmax output, then applies softmax again

**Example:**
```python
model_arch_dict = {
    "num_channels": 1,
    "nb_labels": 5,
    "ndims": 3,
    "nb_levels": 4,
    "nb_features": 24,
    "feat_mult": 2,
    "norm": "batch",
    "final_pred_activation": "softmax"
}
model = UNet(model_arch_dict)
x = torch.randn(2, 1, 128, 128, 128)  # Batch of 2, 1 channel, 128³ volumes
outputs, penultimate = model(x)
# outputs: [2, 5, 128, 128, 128] - final predictions
# penultimate: [2, 5, 128, 128, 128] - before final activation
```

#### Properties

##### `arch_dict`

Returns the model architecture dictionary.

**Returns:**
- `dict`: The model architecture configuration dictionary

---

### `fsdeepnet.models.unet.ConvBlock`

Convolutional block used in U-Net encoder and decoder paths. Supports multiple convolutions per level, residual connections, and various activation functions.

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
- `ndims` (int): Number of dimensions (2 or 3). Determines whether to use `Conv2d` or `Conv3d`. Default: `3`
- `conv_size` (int): Convolution kernel size. Default: `3`
- `nb_conv_per_level` (int): Number of consecutive convolutions in the block. Default: `1`
- `use_residuals` (bool): Whether to use residual/skip connections. If `True`, adds input to output. Default: `False`
- `activation` (str): Activation function - `"elu"` or `"relu"`. Default: `"elu"`
- `weight_init` (str): Weight initialization method - `"xavier_uniform"` or `"zeros"`. Default: `"xavier_uniform"`

**Raises:**
- `ValueError`: If `ndims` is not 2 or 3
- `ValueError`: If `activation` is not `"elu"` or `"relu"`
- `ValueError`: If `weight_init` is not `"xavier_uniform"` or `"zeros"`

**Behavior:**
- Creates `nb_conv_per_level` convolution layers
- Each convolution uses `padding='same'` to preserve spatial dimensions
- Applies activation after each convolution (except the last if residuals are used)
- If `use_residuals=True`, adds a residual connection:
  - Projects input to `out_channels` using convolution with `kernel_size=conv_size` and `padding=1`
  - Adds residual to output before final activation
- Initializes all biases to zero
- Initializes weights using specified method

#### Methods

##### `forward(x)`

Forward pass through the convolutional block.

**Parameters:**
- `x` (torch.Tensor): Input tensor with shape `[N, in_channels, H, W(, D)]`

**Returns:**
- `torch.Tensor`: Output tensor with shape `[N, out_channels, H, W(, D)]` (spatial dimensions preserved)

**Behavior:**
- Applies all convolutions and activations sequentially
- If residuals are enabled, adds input (projected to output channels) to the result
- Returns activated output

**Example:**
```python
# 3D convolutional block with 2 convolutions and residuals
conv_block = ConvBlock(
    in_channels=32,
    out_channels=64,
    ndims=3,
    conv_size=3,
    nb_conv_per_level=2,
    use_residuals=True,
    activation="elu"
)
x = torch.randn(2, 32, 64, 64, 64)
output = conv_block(x)
# output: [2, 64, 64, 64, 64] - same spatial size, more channels
```

---

### `fsdeepnet.models.model_print`

Prints the string representation of a PyTorch model, showing its architecture hierarchy.

#### Function Signature

```python
model_print(model, logger=logging)
```

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to print
- `logger` (logging.Logger, optional): Logger instance for output. If `None` or not provided, uses the `logging` module. Default: `logging`

**Returns:**
- `None`: This function logs output but does not return a value

**Behavior:**
- Calls `logger.info(str(model))` to print the model's `__str__` representation
- The output shows the model architecture in a hierarchical format, displaying all modules and submodules
- Useful for quick inspection of model structure

**Example:**
```python
from fsdeepnet.models import UNet, model_print
import logging

model_config = {
    "num_channels": 1,
    "nb_labels": 50,
    "ndims": 3,
    "nb_levels": 4
}
model = UNet(model_config)
model_print(model, logger=logging)
# Logs the full model architecture hierarchy
```

---

### `fsdeepnet.models.model_summary_torchinfo`

Generates a comprehensive model summary using the `torchinfo` library, including layer information, parameter counts, and memory usage estimates.

#### Function Signature

```python
model_summary_torchinfo(model, input_size)
```

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to summarize
- `input_size` (tuple): Input size tuple for the model. Format depends on dimensionality:
  - **2D models**: `(batch_size, channels, height, width)`
  - **3D models**: `(batch_size, channels, height, width, depth)`
  - Example: `(1, 1, 160, 160, 160)` for a 3D single-channel input with batch size 1

**Returns:**
- `None`: This function prints output but does not return a value

**Raises:**
- `ImportError`: If the `torchinfo` package is not installed

**Behavior:**
- Wrapper around `torchinfo.summary()` for convenience
- Generates a detailed summary including:
  - Layer-by-layer architecture breakdown
  - Input/output shapes for each layer
  - Parameter counts per layer
  - Total parameter count
  - Memory usage estimates (forward pass, backward pass, total)
  - Model size information

**Dependencies:**
- Requires `torchinfo` package: `pip install torchinfo`

**Example:**
```python
from fsdeepnet.models import UNet, model_summary_torchinfo

model_config = {
    "num_channels": 1,
    "nb_labels": 50,
    "ndims": 3,
    "nb_levels": 4,
    "nb_features": 24
}
model = UNet(model_config)
model_summary_torchinfo(model, input_size=(1, 1, 160, 160, 160))
# Prints detailed summary with layer info, parameters, and memory usage
```

---

### `fsdeepnet.models.model_summary`

Generates a detailed model summary using forward hooks to capture input/output shapes, parameter information, and layer details during a dummy forward pass.

#### Function Signature

```python
model_summary(model, input_size, logger=logging, device=None, debug=False)
```

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to summarize
- `input_size` (tuple): Input size tuple **without batch dimension**. Format:
  - **2D models**: `(channels, height, width)`
  - **3D models**: `(channels, height, width, depth)`
  - Example: `(1, 160, 160, 160)` for a 3D single-channel input
  - **Note**: Batch dimension is automatically added as 1 for the dummy forward pass
- `logger` (logging.Logger, optional): Logger instance for output. If `None` or not provided, uses the `logging` module. Default: `logging`
- `device` (torch.device or str, optional): Device for dummy forward pass. If `None`, automatically selects CUDA if available, else CPU. Default: `None`
- `debug` (bool): If `True`, logs detailed hook registration information. Default: `False`

**Returns:**
- `None`: This function logs output but does not return a value

**Behavior:**
1. Registers forward hooks on all leaf modules (modules without children) in the model
2. Runs a dummy forward pass with a zeros tensor of the specified input size
3. Collects information during the forward pass:
   - Input shapes for each layer
   - Output shapes for each layer
   - Parameter counts per layer
   - Parameter sizes (detailed breakdown by parameter name)
4. Removes all hooks after collection
5. Prints a formatted summary table with:
   - Layer name and type
   - Input shape
   - Output shape
   - Parameter count (formatted with commas)
   - Parameter sizes (detailed breakdown)
   - Total parameter count at the end

**Summary Table Format:**
The function prints a table with columns:
- **Layer (type)**: Hierarchical name and class type
- **Input Shape**: Input tensor shape for the layer
- **Output Shape**: Output tensor shape from the layer
- **Param #**: Number of parameters in the layer (formatted with commas)
- **Param Size**: Detailed breakdown of parameter shapes (e.g., `weight:[3, 3, 3]`)

**Notes:**
- Recursively traverses the model to register hooks on all leaf modules
- Builds hierarchical names for nested modules (e.g., `encoder.0.ConvBlock-1`)
- The dummy forward pass uses `torch.zeros()` to avoid any computation overhead
- All hooks are properly removed after the summary is generated

**Example:**
```python
from fsdeepnet.models import UNet, model_summary
import torch

model_config = {
    "num_channels": 1,
    "nb_labels": 50,
    "ndims": 3,
    "nb_levels": 4,
    "nb_features": 24
}
model = UNet(model_config)
model_summary(
    model, 
    input_size=(1, 160, 160, 160),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    debug=False
)
# Prints detailed forward hook summary table
```

---

### `fsdeepnet.models.model_parameters`

Prints all model parameters with their names, shapes, element counts, and trainability status.

#### Function Signature

```python
model_parameters(model, logger=logging)
```

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to inspect
- `logger` (logging.Logger, optional): Logger instance for output. If `None` or not provided, uses the `logging` module. Default: `logging`

**Returns:**
- `None`: This function logs output but does not return a value

**Behavior:**
- Iterates through all named parameters in the model using `model.named_parameters()`
- For each parameter, logs:
  - **Parameter name**: Hierarchical path (e.g., `encoder.0.convs.0.0.weight`)
  - **Shape**: Parameter dimensions as a list (e.g., `[24, 1, 3, 3, 3]`)
  - **Element count**: Total number of elements (formatted with commas)
  - **Trainability**: Whether the parameter requires gradients (`trainable=True/False`)
- Logs total parameter count across all parameters at the end

**Output Format:**
Each parameter is logged as:
```
    {name:30s}: {shape:20s}, {numel:10,d}  trainable={trainable}
```

**Use Cases:**
- Understanding model structure and parameter organization
- Debugging parameter-related issues (e.g., frozen parameters)
- Verifying which parameters are trainable vs. frozen
- Estimating model size and memory requirements
- Inspecting weight initialization

**Example:**
```python
from fsdeepnet.models import UNet, model_parameters

model_config = {
    "num_channels": 1,
    "nb_labels": 50,
    "ndims": 3,
    "nb_levels": 4
}
model = UNet(model_config)
model_parameters(model)
# Logs all parameters with shapes and trainability
# Example output:
#     encoder.0.convs.0.0.weight: [24, 1, 3, 3, 3],     648  trainable=True
#     encoder.0.convs.0.0.bias: [24],                      24  trainable=True
#     ...
# Total parameters: 1,234,567
```

---

### `fsdeepnet.models.model_arch`

Prints the model architecture dictionary in a formatted, readable way.

#### Function Signature

```python
model_arch(arch_dict, logger=logging)
```

**Parameters:**
- `arch_dict` (dict): Model architecture dictionary containing architecture parameters. Typically accessed via `model.arch_dict` property
- `logger` (logging.Logger, optional): Logger instance for output. If `None` or not provided, uses the `logging` module. Default: `logging`

**Returns:**
- `None`: This function logs output but does not return a value

**Behavior:**
- Prints the model name (from `'name'` key if present) as a header
- Logs all architecture parameters with their values in a formatted way
- Each parameter is indented and displayed as `{key}: {value}`
- Prints an empty line at the end for readability

**Output Format:**
```
{model_name}:
    num_channels: 1
    nb_labels: 50
    ndims: 3
    nb_features: 24
    nb_levels: 4
    feat_mult: 2
    ...
```

**Use Cases:**
- Inspecting model configuration and architecture parameters
- Verifying architecture settings after model initialization
- Debugging configuration issues
- Documentation and logging of model architecture
- Comparing different model configurations

**Example:**
```python
from fsdeepnet.models import UNet, model_arch

model_config = {
    "num_channels": 1,
    "nb_labels": 50,
    "ndims": 3,
    "nb_levels": 4,
    "nb_features": 24,
    "feat_mult": 2,
    "norm": "batch"
}
model = UNet(model_config)
model_arch(model.arch_dict)
# Logs:
# UNet:
#     num_channels: 1
#     nb_labels: 50
#     ndims: 3
#     nb_features: 24
#     nb_levels: 4
#     feat_mult: 2
#     norm: batch
#     ...
```

---

## Datasets

### `fsdeepnet.datasets.segmentationdataset.SegmentationDataset`

PyTorch Dataset class for medical image segmentation tasks. Handles loading of images, labels, and optional prior probability maps with support for data augmentation, preloading, and in-memory storage.

#### Constructor

```python
SegmentationDataset(
    augment_obj,
    device=None,
    cohort=[],
    dataset_list_file=None,
    expected_num_channels=1,
    ndims=3,
    num_labels=None,
    label_mapping=None,
    diff_res=True,
    keep_trainset_in_memory=False,
    preload=False,
    augdir=None,
    **kwargs
)
```

**Parameters:**
- `augment_obj`: Augmentation object (e.g., `AugmentBase` instance) for applying data augmentations. Can be `None` if no augmentation is needed
- `device` (torch.device, optional): Device for data loading. If `None`, automatically selects CUDA if available, else CPU
- `cohort` (list): List of cohort names to load from dataset list file (e.g., `["train"]`, `["train", "validation"]`). Default: `[]`
- `dataset_list_file` (str): Path to YAML file containing dataset entries with `image_filepath`, `label_filepath`, and optionally `prior_filepath`
- `expected_num_channels` (int): Expected number of input channels. Default: `1`
- `ndims` (int): Number of dimensions (2 or 3). Default: `3`
- `num_labels` (int, optional): Number of segmentation labels/classes. If `None`, will be determined from data
- `label_mapping` (dict, optional): Dictionary mapping original label IDs to model label indices: `{original_label: model_index}`
- `diff_res` (bool): Whether to allow different resolutions between images and labels. Default: `True`
- `keep_trainset_in_memory` (bool): Whether to keep entire dataset in memory after preloading. Requires `preload=True`. Default: `False`
- `preload` (bool): Whether to preload all data during initialization for validation. Default: `False`
- `augdir` (str, optional): Directory for saving augmented volumes. Cannot be used with `keep_trainset_in_memory=True`
- `**kwargs`: Additional keyword arguments (passed through, may include dataset configuration)

**Raises:**
- `AssertionError`: If `ndims` is not 2 or 3
- `ValueError`: If `keep_trainset_in_memory=True` and `augdir` is not `None`
- `ValueError`: If `keep_trainset_in_memory=True` and `preload=False`
- `AssertionError`: If image and label file lists have different lengths
- `AssertionError`: If label and prior file lists have different lengths

**Behavior:**
- Loads dataset list from YAML file and extracts entries for specified cohorts
- Extracts file paths for images, labels, and priors
- Loads first label to determine target resolution and label lookup table
- If `preload=True`, validates all data and optionally stores in memory
- Creates dataset profile dictionary with metadata
- Validates augmentation object if provided

**Dataset List File Format:**
```yaml
train:
  - image_filepath: /path/to/image1.mgz
    label_filepath: /path/to/label1.mgz
    prior_filepath: /path/to/prior1.mgz  # optional
  - image_filepath: /path/to/image2.mgz
    label_filepath: /path/to/label2.mgz
validation:
  - image_filepath: /path/to/image3.mgz
    label_filepath: /path/to/label3.mgz
```

**Example:**
```python
from fsdeepnet.datasets.segmentationdataset import SegmentationDataset
from fsdeepnet.augmentation import augmentbase

augment_obj = augmentbase.AugmentBase(...)
dataset = SegmentationDataset(
    augment_obj=augment_obj,
    dataset_list_file="data/dataset_list.yaml",
    cohort=["train"],
    expected_num_channels=1,
    ndims=3,
    num_labels=5,
    preload=True,
    keep_trainset_in_memory=True
)
```

#### Methods

##### `__len__()`

Returns the number of samples in the dataset.

**Returns:**
- `int`: Number of dataset entries

##### `__getitem__(index)`

Returns a data sample with optional augmentation applied.

**Parameters:**
- `index` (int): Sample index (0-based)

**Returns:**
- `tuple`: Tuple containing:
  - `index` (int): Dataset entry index
  - `image_tensor` (torch.Tensor or None): Image tensor with shape `[C, H, W(, D)]` (non-batched), or `None` if no images
  - `onehot_label_tensor` (torch.Tensor): One-hot encoded label tensor with shape `[num_classes, H, W(, D)]` (non-batched)
  - `priors_tensor` (torch.Tensor): Prior probability tensor with shape `[num_classes, H, W(, D)]` (non-batched), or empty tensor if no priors

**Raises:**
- `AssertionError`: If label tensor has wrong number of channels
- `AssertionError`: If label resolution differs from target resolution beyond threshold
- `AssertionError`: If image and label shapes don't match
- `AssertionError`: If image and label resolutions don't match (when `diff_res=False`)

**Behavior:**
- If `keep_trainset_in_memory=False`: Loads image, label, and priors from disk
- If `keep_trainset_in_memory=True`: Retrieves preloaded data from memory
- Validates channel counts and resolutions
- Applies data augmentation if `augment_obj` is provided
- Converts labels to one-hot encoding using `label_mapping`
- Returns empty tensor for priors if not available (DataLoader cannot return `None`)

**Example:**
```python
index, image, onehot_label, priors = dataset[0]
# image: [C, H, W, D] or None
# onehot_label: [num_classes, H, W, D]
# priors: [num_classes, H, W, D] or empty tensor
```

##### `hasimage()`

Checks if the dataset contains image files.

**Returns:**
- `bool`: `True` if dataset has image files, `False` otherwise

##### `haspriors()`

Checks if the dataset contains prior probability map files.

**Returns:**
- `bool`: `True` if dataset has prior files, `False` otherwise

##### `preload()`

Preprocesses all label maps, validates data, and optionally stores data in memory. Called automatically during initialization if `preload=True`.

**Returns:**
- `tuple`: Tuple containing:
  - `generation_labels` (set): Set of all unique label values found in the dataset
  - `res_diffs` (int): Number of images/labels with resolution differences beyond threshold

**Behavior:**
- Loads and validates all images, labels, and priors
- Checks shapes, channels, and resolutions
- Collects all unique labels from the dataset
- If `keep_trainset_in_memory=True`, stores all data in memory
- Logs dataset information and validation results
- Exits with error if resolution differences exceed threshold

**Raises:**
- `AssertionError`: If any validation check fails (shape, channels, resolution)
- `SystemExit`: If `res_diffs > 0` after validation

##### `SegmentationDataset.process_dataset_attr(dataset_profile, traindir)`

Static method to process and update dataset attributes. Handles loading of label arrays from files and creates label mappings.

**Parameters:**
- `dataset_profile` (dict): Dataset profile dictionary containing configuration
- `traindir` (str): Training output directory for saving processed label arrays

**Returns:**
- `dict`: Updated dataset profile dictionary

**Behavior:**
- Processes label arrays that can be provided as:
  - File paths (strings): Loads from `.npy` files
  - Arrays/lists: Uses directly
- Creates `label_mapping` and `inverse_label_mapping` from `segmentation_labels`
- Saves label arrays to `.npy` files in `traindir` if provided:
  - `segmentation_labels.npy`
  - `segmentation_names.npy`
  - `generation_labels.npy`
  - `generation_classes.npy`
  - `topology_classes.npy`
- Updates dataset profile with processed attributes

**Keys Processed:**
- `generation_labels`: Labels used for generation
- `generation_classes`: Classes for generation
- `segmentation_labels`: Segmentation label IDs
- `segmentation_names`: Label names
- `topology_classes`: Topology class assignments
- `parcellation_labels`: Parcellation label IDs
- `parcellation_names`: Parcellation label names

**Example:**
```python
dataset_profile = {
    "segmentation_labels": [0, 2, 3, 4, 17, 41],
    "segmentation_names": ["Background", "Left-Hippocampus", ...]
}
updated_profile = SegmentationDataset.process_dataset_attr(
    dataset_profile, 
    traindir="output/training"
)
# Creates label_mapping, inverse_label_mapping, num_labels, etc.
```

#### Properties

##### `profile`

Returns the processed dataset profile dictionary.

**Returns:**
- `dict`: Dataset profile containing:
  - `num_samples`: Number of dataset entries
  - `input_shape`: Input tensor shape `[C, H, W(, D)]`
  - `target_res`: Target resolution `[x, y(, z)]` in mm
  - `res_diff_thresh`: Resolution difference threshold
  - `hasimage`: Whether dataset has images
  - `haspriors`: Whether dataset has priors
  - `label_lookup`: Label lookup table
  - `reported_generation_labels`: All unique labels found (if preloaded)

---

## Augmentation

### `fsdeepnet.augmentation.apply_augmentations`

Applies a sequence of augmentations to image and label tensors in the order specified by the augmentation object.

#### Function Signature

```python
apply_augmentations(
    augment_obj,
    image_tensor,
    label_tensor,
    original_image,
    original_label,
    priors_tensor=None,
    orig_fpath=None,
    index=None
)
```

**Parameters:**
- `augment_obj` (AugmentBase or AugmentVoxynth): Augmentation object containing the augmentation pipeline configuration. Must have:
  - `transforms` attribute: List of augmentation names to apply in order
  - `output_dir` attribute: Optional directory for saving debug volumes
- `image_tensor` (torch.Tensor or None): Input image tensor with shape `[C, H, W(, D)]`. Can be `None` if only labels are being augmented
- `label_tensor` (torch.Tensor): Input label tensor with shape `[C, H, W(, D)]`. Required
- `original_image` (surfa.Volume): Original image volume (before reorientation/preprocessing). Used for saving debug volumes with correct geometry
- `original_label` (surfa.Volume): Original label volume (before reorientation/preprocessing). Used for saving debug volumes with correct geometry
- `priors_tensor` (torch.Tensor, optional): Optional prior probability tensor with shape `[C, H, W(, D)]`. Default: `None`
- `orig_fpath` (str, optional): Original file path for naming debug output files. If `None` and `augment_obj.output_dir` is set, debug saving is skipped. Default: `None`
- `index` (int, optional): Optional index for naming debug output files (e.g., for batch processing). If provided, files are prefixed with `"{index+1:04d}."`. Default: `None`

**Returns:**
- `tuple`: Tuple containing three tensors:
  - `image_tensor` (torch.Tensor or None): Augmented image tensor with shape `[C, H, W(, D)]`, or `None` if input was `None` or removed by augmentations
  - `label_tensor` (torch.Tensor): Augmented label tensor with shape `[C, H, W(, D)]`
  - `priors_tensor` (torch.Tensor or None): Augmented prior tensor with shape `[C, H, W(, D)]`, or `None` if input was `None` or removed by augmentations

**Behavior:**
1. **Optional Debug Saving**: If `output_dir` and `orig_fpath` are provided, saves reoriented input volumes:
   - Saves as `.mgz` files (with geometry) and `.npy` files (raw arrays)
   - Files prefixed with volume name and optional index
2. **Augmentation Pipeline**: Iterates through augmentations in `augment_obj.transforms` order:
   - Each augmentation is called with a dictionary containing:
     - `'image'`: image_tensor (or None)
     - `'label'`: label_tensor
     - `'prior'`: priors_tensor (or None)
     - `'geom'`: geometry object (updated after spatial augmentations)
   - Output from one augmentation becomes input to the next
3. **Intermediate Debug Saving**: After each augmentation (if debug saving enabled):
   - Saves augmented volumes with augmentation name and index in filename
   - Both `.mgz` (with updated geometry) and `.npy` formats
4. **Geometry Updates**: The geometry object is updated after each augmentation that modifies spatial properties (e.g., resampling, cropping)

**Notes:**
- If an augmentation is not found in `augment_obj`, a warning is logged and it is skipped
- Debug volumes are saved with the format: `{prefix}_{augment_name}_{idx}_{type}.{ext}`
- The function handles cases where augmentations may remove images (e.g., conditional GMM generation)
- All augmentations are applied sequentially; order matters for the final result

**Example:**
```python
from fsdeepnet.augmentation import AugmentBase, apply_augmentations
import torch

# Create augmentation object
augment_obj = AugmentBase(
    hp={
        "flip": {"flip_prob": 0.5},
        "spatialdeformation": {"max_rotation": 15},
        "intensityaugmentation": {"normalize": True}
    },
    transforms=["flip", "spatialdeformation", "intensityaugmentation"],
    augmentation_dir="./debug_aug"
)

# Apply augmentations
aug_image, aug_label, aug_prior = apply_augmentations(
    augment_obj,
    image_tensor=img_tensor,  # [C, H, W, D]
    label_tensor=label_tensor,  # [C, H, W, D]
    original_image=orig_img_vol,  # surfa.Volume
    original_label=orig_label_vol,  # surfa.Volume
    priors_tensor=priors_tensor,  # [C, H, W, D] or None
    orig_fpath="/path/to/image.mgz",
    index=0
)
# Returns augmented tensors with same shapes
```

---

### `fsdeepnet.augmentation.check_augmentations`

Validates augmentation configuration and checks for conflicts between incompatible augmentations.

#### Function Signature

```python
check_augmentations(augment_obj)
```

**Parameters:**
- `augment_obj` (AugmentBase or AugmentVoxynth): Augmentation object to validate. Must have:
  - `transforms` attribute: List of augmentation names to apply
  - `valid_augmentations` attribute: List of supported augmentation names

**Returns:**
- `None`: This function raises exceptions on validation failure but does not return a value

**Raises:**
- `AssertionError`: If any augmentation in `transforms` is not in `valid_augmentations`
- `ValueError`: If conflicting cropping augmentations are selected simultaneously:
  - `'centroidcrop'` conflicts with: `'centercrop'`, `'randomcrop'`, `'randomcentercrop'`
  - `'centercrop'` conflicts with: `'randomcrop'`, `'randomcentercrop'`
  - `'randomcrop'` conflicts with: `'randomcentercrop'`

**Behavior:**
1. **Validation Check**: Verifies that all augmentations in `transforms` are present in `valid_augmentations` list
2. **Conflict Detection**: Checks for mutually exclusive cropping augmentations:
   - Only one cropping operation should be applied per sample
   - Multiple cropping augmentations would result in unpredictable behavior
   - The function detects all pairwise conflicts between cropping types

**Notes:**
- This function should be called before applying augmentations to catch configuration errors early
- The conflict checking is specific to cropping augmentations, as they are mutually exclusive
- Other augmentations can be combined freely (e.g., flip + spatial deformation + intensity augmentation)

**Example:**
```python
from fsdeepnet.augmentation import AugmentBase, check_augmentations

# Valid configuration
augment_obj = AugmentBase(
    hp={},
    transforms=["flip", "spatialdeformation", "intensityaugmentation"]
)
check_augmentations(augment_obj)  # Passes

# Invalid: conflicting cropping augmentations
augment_obj = AugmentBase(
    hp={},
    transforms=["centercrop", "randomcrop"]  # Conflict!
)
try:
    check_augmentations(augment_obj)
except ValueError as e:
    print(f"Error: {e}")
    # Error: Both 'centercrop' and 'randomcrop' are selected. Choose one.

# Invalid: unknown augmentation
augment_obj = AugmentBase(
    hp={},
    transforms=["unknown_augmentation"]  # Not in valid_augmentations
)
try:
    check_augmentations(augment_obj)
except AssertionError as e:
    print(f"Error: {e}")
    # Error: Unknown augmentation 'unknown_augmentation'. Supported augmentations [...]
```

---

### `fsdeepnet.augmentation.augmentbase.AugmentBase`

Wrapper class for medical image data augmentation. Manages multiple augmentation transforms and applies them sequentially to images, labels, and optional prior probability maps.

#### Class Variables

- `RES_DIFF_THRESH` (float): Resolution difference threshold (default: `0.05`, i.e., 5%). Used to determine if resampling is necessary.

#### Constructor

```python
AugmentBase(
    hp,
    transforms,
    crop_size=None,
    augmentation_dir=None,
    device=None,
    target_res=None,
    num_channels=1,
    sampling_hp=True,
    verbose=False,
    left_right_corresponding=None,
    bbox_labels=None,
    generation_labels=None,
    generation_classes=None,
    segmentation_labels=None,
    **kwargs
)
```

**Parameters:**
- `hp` (dict): Hyperparameters dictionary. Keys should match augmentation names (e.g., `"flip"`, `"spatialdeformation"`). Each value is a dict of hyperparameters for that augmentation
- `transforms` (list): List of augmentation names to apply in sequence. Must be from `valid_augmentations`
- `crop_size` (list, optional): Crop size `[W, H(, D)]` for cropping augmentations. Default: `None`
- `augmentation_dir` (str, optional): Output directory for saving augmented volumes (for debugging). Default: `None`
- `device` (torch.device, optional): Device for augmentation operations. If `None`, automatically selects CUDA if available, else CPU
- `target_res` (list, optional): Target resolution `[x, y(, z)]` in mm for resampling. Default: `None`
- `num_channels` (int): Number of image channels. Default: `1`
- `sampling_hp` (bool): Whether to sample hyperparameters from ranges (for stochastic augmentations). Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`
- `left_right_corresponding` (list, optional): List of left-right label pairs `[left1, right1, left2, right2, ...]` for flip augmentation
- `bbox_labels` (list, optional): Label values to use for bounding box computation in cropping
- `generation_labels` (array, optional): Label IDs for conditional GMM generation
- `generation_classes` (array, optional): Class assignments for generation labels
- `segmentation_labels` (array, optional): Target segmentation labels for remapping
- `**kwargs`: Additional keyword arguments (passed through)

**Valid Augmentations:**
- `"flip"`: Left-right flipping with label swapping
- `"spatialdeformation"`: Affine and non-linear spatial deformations
- `"randomcrop"`: Random cropping (optionally around bounding box)
- `"randomcentercrop"`: Random center cropping (optionally around bounding box)
- `"centercrop"`: Center cropping with optional random offset
- `"centroidcrop"`: Cropping centered on label centroid
- `"biasfieldcorruption"`: Multiplicative bias field corruption
- `"intensityaugmentation"`: Intensity augmentation (noise, gamma scaling, normalization)
- `"sampleconditionalgmm"`: Conditional GMM sampling for synthetic image generation
- `"rescalevolume"`: Intensity rescaling with percentile clipping
- `"gaussianblur"`: Gaussian blurring
- `"resamplevolume"`: Resampling to target resolution
- `"mimicresolution"`: Simulating low-resolution acquisition
- `"remaplabels"`: Label remapping

**Behavior:**
- Initializes individual augmentation module instances for each supported augmentation
- Stores augmentation sequence in `self.transforms`
- Sets `self.output_dir` for debug output
- Validates augmentation names against `valid_augmentations`

**Example:**
```python
hp = {
    "flip": {"flip_prob": 0.5},
    "spatialdeformation": {"affine_probability": 1.0, "max_translation": 5.0},
    "intensityaugmentation": {"normalize": True, "gamma_scaling_max": 0.5}
}
transforms = ["flip", "spatialdeformation", "intensityaugmentation"]
augment_obj = AugmentBase(
    hp=hp,
    transforms=transforms,
    crop_size=[160, 160, 160],
    left_right_corresponding=[2, 41, 3, 42],  # Left/Right hippocampus
    device=torch.device("cuda")
)
```

#### Properties

##### `valid_augmentations`

List of valid augmentation names supported by this class.

**Returns:**
- `list`: List of augmentation name strings

##### `output_dir`

Output directory for saving augmented volumes (for debugging).

**Returns:**
- `str` or `None`: Directory path, or `None` if not set

---

### Individual Augmentation Classes

All augmentation classes inherit from `torch.nn.Module` and implement a `forward()` method that takes an input dictionary and returns an output dictionary.

#### Input/Output Format

**Input Dictionary:**
```python
{
    "image": torch.Tensor,      # Shape [C, H, W(, D)], optional
    "label": torch.Tensor,      # Shape [1, H, W(, D)], required
    "prior": torch.Tensor,      # Shape [num_classes, H, W(, D)], optional
    "geom": surfa.ImageGeometry, # Image geometry, optional
    "crop_idx": torch.Tensor,   # Crop indices, optional
    ...
}
```

**Output Dictionary:**
```python
{
    "image": torch.Tensor,      # Augmented image (same shape)
    "label": torch.Tensor,      # Augmented label (same shape)
    "prior": torch.Tensor,      # Augmented prior (same shape), optional
    "geom": surfa.ImageGeometry, # Updated geometry, optional
    "crop_idx": torch.Tensor,   # Crop indices, optional
    ...
}
```

---

### `fsdeepnet.augmentation.augmentbase.Flip`

Left-right flipping augmentation with automatic label swapping for symmetric structures.

#### Constructor

```python
Flip(left_right_corresponding, hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `left_right_corresponding` (list): List of left-right label pairs `[left1, right1, left2, right2, ...]`
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  - `flip_prob` (float): Probability of applying flip. Default: `0.5`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

#### Methods

##### `forward(input, **kwargs)`

Applies random left-right flip to image and label volumes, swapping left-right labels.

**Parameters:**
- `input` (dict): Input dictionary with `image`, `label`, `geom`, etc.

**Returns:**
- `dict`: Output dictionary with flipped image and label

**Raises:**
- `AssertionError`: If `geom` is `None` when flip is applied
- `AssertionError`: If `left_right_corresponding` is `None` when flip is applied

**Behavior:**
- Randomly applies flip based on `flip_prob`
- Swaps left and right labels according to `left_right_corresponding`
- Finds left-right axis from image geometry (RAS orientation)
- Flips image and label tensors along the left-right axis

---

### `fsdeepnet.augmentation.augmentbase.SpatialDeformation`

Affine and non-linear spatial deformation augmentation using velocity fields.

#### Constructor

```python
SpatialDeformation(hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  - `affine_probability` (float): Probability of applying affine transformation. Default: `1.0`
  - `max_translation` (float): Maximum translation in mm. Default: `5.0`
  - `max_rotation` (float): Maximum rotation in degrees. Default: `5.0`
  - `max_shearing` (float): Maximum shearing factor. Default: `0.015`
  - `max_scaling` (float): Maximum scaling factor. Default: `1.1`
  - `warp_probability` (float): Probability of applying non-linear warp. Default: `1.0`
  - `warp_integrations` (int): Number of integration steps for velocity field. Default: `7`
  - `warp_generation_method` (str): Warp generation method - `"gaussian"` or `"perlin"`. Default: `"perlin"`
  - `warp_perlin_method` (str): Perlin noise method - `"upsample"` or other. Default: `"upsample"`
  - `warp_smoothing_range` (list): Smoothing range for Perlin method `[min, max]`. Default: `[10, 20]`
  - `warp_magnitude_range` (list): Magnitude range for Perlin method `[min, max]`. Default: `[1, 2]`
  - `warp_nonlin_scale` (float): Scale for Gaussian method. Default: `0.04`
  - `warp_nonlin_std` (float): Standard deviation for Gaussian method. Default: `4.0`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

**Raises:**
- `AssertionError`: If `warp_generation_method` is not `"gaussian"` or `"perlin"`

#### Methods

##### `forward(input, debugsaveprefix=None, **kwargs)`

Applies random spatial transformation (affine + non-linear warp) to image and label volumes.

**Parameters:**
- `input` (dict): Input dictionary with `image`, `label`, `prior`, `geom`
- `debugsaveprefix` (str, optional): Prefix for saving debug warp files

**Returns:**
- `dict`: Output dictionary with transformed image, label, and prior

**Behavior:**
- Generates random transform using `voxynth.transform.random_transform()`
- Applies transform to image (bilinear interpolation) and label/prior (nearest neighbor)
- Optionally saves warp field and affine matrix for debugging

---

### `fsdeepnet.augmentation.augmentbase.RandomCrop`

Random cropping augmentation with optional bounding box constraints.

#### Constructor

```python
RandomCrop(crop_size, bbox_labels=None, hp=None, device=None, mode='random', sampling_hp=True, verbose=False)
```

**Parameters:**
- `crop_size` (list): Crop size `[W, H(, D)]`
- `bbox_labels` (list, optional): Label values to compute bounding box for. If provided, crop is constrained to include bounding box
- `hp` (dict, optional): Hyperparameters dictionary (currently unused). Default: `{}`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `mode` (str): Cropping mode - `"random"` or `"center"`. Default: `"random"`
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

#### Methods

##### `forward(input, **kwargs)`

Randomly crops input tensors to specified size.

**Parameters:**
- `input` (dict): Input dictionary with `image`, `label`, `prior`, `geom`

**Returns:**
- `dict`: Output dictionary with cropped tensors and `crop_idx` tensor `[start_x, start_y(, start_z), end_x, end_y(, end_z)]`

**Behavior:**
- If `bbox_labels` provided, computes bounding box and constrains crop to include it
- If `mode="random"`, samples random start coordinates
- If `mode="center"`, samples random center point and crops around it
- Returns cropped tensors and crop indices

---

### `fsdeepnet.augmentation.augmentbase.CenterCrop`

Center cropping with optional random offset.

#### Constructor

```python
CenterCrop(crop_size, hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `crop_size` (list): Crop size `[W, H(, D)]`
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  - `max_offset` (list, optional): Maximum offset from center `[x, y(, z)]`. If `None`, crops exactly at center
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample offset randomly. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

#### Methods

##### `forward(input, **kwargs)`

Crops volume centered at image center (with optional random offset).

**Parameters:**
- `input` (dict): Input dictionary

**Returns:**
- `dict`: Output dictionary with cropped tensors and `crop_idx`

**Behavior:**
- Computes image center
- If `max_offset` provided, adds random offset (uniformly sampled if `sampling_hp=True`)
- Crops around center point
- Returns original volume if crop size >= image size

---

### `fsdeepnet.augmentation.augmentbase.CentroidCrop`

Cropping centered on label centroid.

#### Constructor

```python
CentroidCrop(crop_size, hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `crop_size` (list): Crop size `[W, H(, D)]`
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  - `max_offset` (list, optional): Maximum offset from centroid `[x, y(, z)]`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample offset randomly. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

#### Methods

##### `forward(input, **kwargs)`

Crops volume centered at label centroid (non-zero labels).

**Parameters:**
- `input` (dict): Input dictionary with `label` (required)

**Returns:**
- `dict`: Output dictionary with cropped tensors and `crop_idx`

**Behavior:**
- Computes centroid of non-zero labels
- If `max_offset` provided, adds random offset
- Adjusts center to ensure full crop fits within image
- Returns original volume if crop size >= image size

---

### `fsdeepnet.augmentation.augmentbase.IntensityAugmentation`

Intensity augmentation with noise injection, normalization, and gamma scaling.

#### Constructor

```python
IntensityAugmentation(hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  - `added_noise_max_sigma` (float): Maximum standard deviation for Gaussian noise. Default: `1.0` (0 = disabled)
  - `added_noise_probability` (float): Probability of applying noise. Default: `0.95`
  - `clip_values` (list or float, optional): Clipping range `[min, max]` or single value (clips to `[0, value]`). Default: `None`
  - `normalize` (bool): Whether to apply min-max normalization to `[0, 1]`. Default: `True`
  - `gamma_scaling_max` (float): Maximum standard deviation for gamma distribution. Default: `0.5` (0 = disabled)
  - `gamma_scaling_probability` (float): Probability of applying gamma scaling. Default: `1.0`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

#### Methods

##### `forward(input, **kwargs)`

Applies intensity augmentation: noise → clipping → normalization → gamma scaling.

**Parameters:**
- `input` (dict): Input dictionary with `image` (required)

**Returns:**
- `dict`: Output dictionary with augmented image

**Behavior:**
- Adds Gaussian noise (per channel) if enabled
- Clips values if `clip_values` specified
- Normalizes to `[0, 1]` if `normalize=True`
- Applies gamma scaling (voxel-wise exponentiation) if enabled
- All operations applied independently per channel

---

### `fsdeepnet.augmentation.augmentbase.BiasFieldCorruption`

Multiplicative bias field corruption to simulate MRI intensity inhomogeneities.

#### Constructor

```python
BiasFieldCorruption(hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  - `bias_field_max_magnitude` (float): Maximum magnitude for bias field. Default: `0.7`
  - `bias_field_scale` (float): Scale ratio between image and small bias field. Default: `0.025`
  - `bias_field_probability` (float): Probability of applying bias field. Default: `0.95`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

#### Methods

##### `forward(input, **kwargs)`

Applies smooth multiplicative bias field to image.

**Parameters:**
- `input` (dict): Input dictionary with `image` (required)

**Returns:**
- `dict`: Output dictionary with corrupted image

**Behavior:**
- Samples small low-resolution velocity field from normal distribution
- Upsamples to image size using trilinear/bilinear interpolation
- Takes exponential to ensure non-negativity
- Multiplies bias field with image (per channel)

---

### `fsdeepnet.augmentation.augmentbase.SampleConditionalGMM`

Generates synthetic images by sampling a Gaussian Mixture Model conditioned on labels.

#### Constructor

```python
SampleConditionalGMM(generation_labels, generation_classes, hp=None, num_channels=1, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `generation_labels` (array): Label IDs used for generation
- `generation_classes` (array): Class assignments for each generation label
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  - `prior_distribution` (str): Distribution for sampling GMM parameters - `"uniform"` or `"normal"`. Default: `"uniform"`
  - `prior_mean` (list): Mean hyperparameters `[min, max]` for uniform or `[mean, std]` for normal. Default: `[25, 225]`
  - `prior_std` (list): Standard deviation hyperparameters `[min, max]` for uniform or `[mean, std]` for normal. Default: `[5, 25]`
- `num_channels` (int): Number of output channels. Default: `1`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

#### Methods

##### `forward(input, **kwargs)`

Generates synthetic image by sampling GMM conditioned on label map.

**Parameters:**
- `input` (dict): Input dictionary with `label` (required)

**Returns:**
- `dict`: Output dictionary with generated `image` and original `label`

**Raises:**
- `AssertionError`: If `label` is `None`
- `AssertionError`: If `generation_labels` is `None`
- `AssertionError`: If `generation_classes` is `None`
- `ValueError`: If `prior_distribution` is not `"uniform"` or `"normal"`

**Behavior:**
- Samples means and standard deviations for each class from prior distribution
- Applies special handling for background class (5% chance to set to 0, 30% chance to set to low values)
- For each generation label, samples Gaussian values at corresponding voxels
- Each channel sampled independently

---

### `fsdeepnet.augmentation.augmentbase.RescaleVolume`

Intensity rescaling with percentile-based clipping.

#### Constructor

```python
RescaleVolume(hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  - `new_min` (float): Minimum value after rescaling. Default: `0.0`
  - `new_max` (float): Maximum value after rescaling. Default: `1.0`
  - `min_percentile` (float): Percentile for minimum clipping. Default: `0.5`
  - `max_percentile` (float): Percentile for maximum clipping. Default: `99.5`
  - `use_positive_only` (bool): Whether to use only positive values for percentile calculation. Default: `False`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

#### Methods

##### `forward(input, **kwargs)`

Rescales image intensities using percentile-based clipping.

**Parameters:**
- `input` (dict): Input dictionary with `image` (required)

**Returns:**
- `dict`: Output dictionary with rescaled image

**Behavior:**
- Computes percentiles (or min/max if percentiles are 0/100)
- Clips image to percentile range
- Rescales to `[new_min, new_max]`
- Applied independently per channel

---

### `fsdeepnet.augmentation.augmentbase.GaussianBlur`

Gaussian blurring augmentation.

#### Constructor

```python
GaussianBlur(hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  - `gaussian_blur_max_sigma` (float or list): Maximum sigma for Gaussian kernel. Default: `None`
  - `gaussian_blur_range` (float): Range multiplier for randomizing sigma `[1/range, range]`. Default: `None`
  - `gaussian_blur_truncate` (float): Truncate kernel at this many standard deviations. Default: `2.5`
  - `gaussian_blur_radius` (int or list, optional): Explicit kernel radius. If provided, `truncate` is ignored
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

#### Methods

##### `forward(input, sigma, **kwargs)`

Applies Gaussian blur to image.

**Parameters:**
- `input` (dict): Input dictionary with `image` (required)
- `sigma` (float or list): Standard deviation(s) for Gaussian kernel

**Returns:**
- `dict`: Output dictionary with blurred image

**Behavior:**
- Randomizes sigma if `blur_range` provided
- Generates Gaussian kernel using `Filter.gaussian_kernel()`
- Applies convolution with `padding='same'` to preserve shape
- Each channel blurred independently

---

### `fsdeepnet.augmentation.augmentbase.ResampleVolume`

Resamples volume to target resolution.

#### Constructor

```python
ResampleVolume(target_res, resample_thresh=None, hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `target_res` (float or list): Target resolution `[x, y(, z)]` in mm. If scalar, applies to all dimensions
- `resample_thresh` (float, optional): Threshold for resampling (relative difference). If `None`, uses `AugmentBase.RES_DIFF_THRESH`
- `hp` (dict, optional): Hyperparameters dictionary (currently unused). Default: `{}`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

#### Methods

##### `forward(input, **kwargs)`

Resamples image and label to target resolution.

**Parameters:**
- `input` (dict): Input dictionary with `image`, `label`, `geom` (required)

**Returns:**
- `dict`: Output dictionary with resampled tensors and updated geometry

**Behavior:**
- Skips resampling if current resolution is within threshold of target
- Creates sampling grid for target resolution
- Resamples image with bilinear interpolation
- Resamples label with nearest neighbor interpolation
- Updates geometry with new resolution

---

### `fsdeepnet.augmentation.augmentbase.MimicResolution`

Simulates low-resolution acquisition by downsampling and upsampling.

#### Constructor

```python
MimicResolution(hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  - `mimic_probability` (float): Probability of applying resolution mimicking. Default: `1.0`
  - `isotropic_probability` (float): Probability of sampling isotropic resolution (if both iso/aniso specified). Default: `0.1`
  - `min_res_probability` (float): Probability of using original resolution. Default: `0.05`
  - `max_res_iso` (float or list, optional): Maximum isotropic resolution (all dimensions equal)
  - `max_res_aniso` (float or list, optional): Maximum anisotropic resolution (one dimension different)
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

**Raises:**
- `Exception`: If `isotropic_probability=0` but both `max_res_iso` and `max_res_aniso` are specified

#### Methods

##### `forward(input, **kwargs)`

Simulates low-resolution acquisition.

**Parameters:**
- `input` (dict): Input dictionary with `image`, `geom` (required)

**Returns:**
- `dict`: Output dictionary with resolution-mimicked image

**Behavior:**
- Samples random resolution between input resolution and `max_res_iso`/`max_res_aniso`
- Applies Gaussian blur before downsampling
- Downsamples to sampled resolution
- Upsamples back to original resolution
- Returns image with simulated low-resolution appearance

---

### `fsdeepnet.augmentation.augmentbase.RemapLabels`

Remaps label values from source labels to destination labels.

#### Constructor

```python
RemapLabels(source_labels, dest_labels=None, hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `source_labels` (array): Source label IDs to remap from
- `dest_labels` (array, optional): Destination label IDs to remap to. If `None`, maps to sequential indices `[0, 1, 2, ...]`
- `hp` (dict, optional): Hyperparameters dictionary (currently unused). Default: `{}`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

#### Methods

##### `forward(input, **kwargs)`

Remaps label values according to mapping.

**Parameters:**
- `input` (dict): Input dictionary with `label` (required)

**Returns:**
- `dict`: Output dictionary with remapped label

**Behavior:**
- Creates mapping dictionary from source to destination labels
- If source and destination are identical, returns input unchanged
- Applies mapping to label tensor
- Unmapped labels remain unchanged (set to 0)

---

### Utility Functions

#### `fsdeepnet.augmentation.augmentbase.CropVolume(volume, crop_idx, verbose=False)`

Crops a volume tensor using crop indices.

**Parameters:**
- `volume` (torch.Tensor): Volume tensor with shape `[C, H, W(, D)]` (non-batched)
- `crop_idx` (torch.Tensor): Crop indices `[start_x, start_y(, start_z), end_x, end_y(, end_z)]`
- `verbose` (bool): Enable verbose logging. Default: `False`

**Returns:**
- `torch.Tensor`: Cropped volume tensor

**Behavior:**
- Returns original volume if `crop_idx` is `None`
- Supports both 2D and 3D volumes

#### `fsdeepnet.augmentation.augmentbase.PadVolume(volume, padding_shape, padding_value=0)`

Pads a volume tensor to a target shape.

**Parameters:**
- `volume` (torch.Tensor): Volume tensor with shape `[C, H, W(, D)]` (non-batched)
- `padding_shape` (list): Target shape `[H, W(, D)]` (excluding channels)
- `padding_value` (float): Value to use for padding. Default: `0`

**Returns:**
- `tuple`: Tuple containing:
  - `padded_volume` (torch.Tensor): Padded volume tensor
  - `pad_idx` (numpy.ndarray): Padding indices `[start_x, start_y(, start_z), end_x, end_y(, end_z)]`

**Behavior:**
- Pads symmetrically (equal margins on both sides)
- Returns original volume and indices if no padding needed
- Uses `torch.nn.functional.pad()` with constant padding

---

### `fsdeepnet.augmentation.augmentvoxynth.AugmentVoxynth`

Extended augmentation class using the Voxynth library (https://github.com/dalcalab/voxynth/). Inherits from `AugmentBase` and adds additional augmentations that use Voxynth's `image_augment()` function for bias field corruption and intensity augmentation.

#### Constructor

```python
AugmentVoxynth(
    hp,
    transforms,
    crop_size=None,
    augmentation_dir=None,
    device=None,
    target_res=None,
    num_channels=1,
    sampling_hp=True,
    verbose=False,
    left_right_corresponding=None,
    bbox_labels=None,
    generation_labels=None,
    generation_classes=None,
    segmentation_labels=None,
    **kwargs
)
```

**Parameters:**
- `hp` (dict): Hyperparameters dictionary. Keys should match augmentation names. Supports all `AugmentBase` augmentations plus Voxynth-specific ones
- `transforms` (list): List of augmentation names to apply. Can include both base and Voxynth augmentations
- `crop_size` (list, optional): Crop size `[W, H(, D)]` for cropping augmentations. Default: `None`
- `augmentation_dir` (str, optional): Output directory for saving augmented volumes. Default: `None`
- `device` (torch.device, optional): Device for augmentation operations. If `None`, automatically selects CUDA if available, else CPU
- `target_res` (list, optional): Target resolution `[x, y(, z)]` in mm. Default: `None`
- `num_channels` (int): Number of image channels. Default: `1`
- `sampling_hp` (bool): Whether to sample hyperparameters from ranges. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`
- `left_right_corresponding` (list, optional): List of left-right label pairs for flip augmentation
- `bbox_labels` (list, optional): Label values for bounding box computation
- `generation_labels` (array, optional): Label IDs for conditional GMM generation
- `generation_classes` (array, optional): Class assignments for generation labels
- `segmentation_labels` (array, optional): Target segmentation labels for remapping
- `**kwargs`: Additional keyword arguments (passed to parent class)

**Additional Valid Augmentations:**
- `"biasfieldcorruption"`: Voxynth-based bias field corruption (overrides base class version)
- `"intensityaugmentation"`: Voxynth-based intensity augmentation (overrides base class version)
- `"biasfieldcorruptionandintensityaugmentation"`: Combined bias field and intensity augmentation in a single call

**Behavior:**
- Inherits all functionality from `AugmentBase`
- Extends `valid_augmentations` list with Voxynth-specific augmentations
- Initializes Voxynth-based augmentation instances that use `voxynth.augment.image_augment()`
- All base class augmentations remain available

**Example:**
```python
hp = {
    "biasfieldcorruption": {
        "bias_field_probability": 0.5,
        "bias_field_max_magnitude": 0.1,
        "bias_field_scale": 0.025,
        "bias_field_generation_method": "blur"
    },
    "intensityaugmentation": {
        "added_noise_probability": 0.5,
        "added_noise_max_sigma": 0.05,
        "gamma_scaling_probability": 0.5,
        "gamma_scaling_max": 0.8
    }
}
transforms = ["biasfieldcorruption", "intensityaugmentation"]
augment_obj = AugmentVoxynth(
    hp=hp,
    transforms=transforms,
    device=torch.device("cuda")
)
```

---

### Voxynth-Specific Augmentation Classes

These classes use the Voxynth library's `image_augment()` function and provide alternative implementations to the base class versions.

---

### `fsdeepnet.augmentation.augmentvoxynth.BiasFieldCorruption`

Voxynth-based bias field corruption augmentation. Uses `voxynth.augment.image_augment()` for bias field generation.

#### Constructor

```python
BiasFieldCorruption(hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  - `bias_field_probability` (float): Probability of applying bias field. Default: `0.5`
  - `bias_field_max_magnitude` (float): Maximum magnitude for bias field. Default: `0.1`
  - `bias_field_smoothing_range` (list, optional): Smoothing range `[min, max]` for bias field. If `None`, calculated from `bias_field_scale`. Default: `None`
  - `bias_field_scale` (float): Scale ratio between image and small bias field. Default: `0.025`
  - `bias_field_generation_method` (str): Generation method - `"blur"` or `"upsample"`. Default: `"blur"`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

**Raises:**
- `AssertionError`: If `bias_field_generation_method` is not `"blur"` or `"upsample"`

**Behavior:**
- If `bias_field_smoothing_range` is `None`, calculates it from `bias_field_scale`:
  - For `"blur"` method: Uses Gaussian standard deviation based on FWHM
  - For `"upsample"` method: Uses voxel size directly
- Uses `voxynth.augment.image_augment()` for bias field generation
- Applies multiplicative bias field to image

#### Methods

##### `forward(input, **kwargs)`

Applies bias field corruption using Voxynth library.

**Parameters:**
- `input` (dict): Input dictionary with `image`, `geom` (required)

**Returns:**
- `dict`: Output dictionary with corrupted image

**Behavior:**
- Calls `voxynth.augment.image_augment()` with bias field parameters
- Copies `geom.voxsize` to make it writable (required by Voxynth)
- Returns augmented image with unchanged label, prior, and geometry

---

### `fsdeepnet.augmentation.augmentvoxynth.IntensityAugmentation`

Voxynth-based intensity augmentation. Uses `voxynth.augment.image_augment()` for noise and gamma scaling.

#### Constructor

```python
IntensityAugmentation(hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  - `added_noise_probability` (float): Probability of applying noise. Default: `0.5`
  - `added_noise_max_sigma` (float): Maximum standard deviation for Gaussian noise. Default: `0.05`
  - `gamma_scaling_probability` (float): Probability of applying gamma scaling. Default: `0.5`
  - `gamma_scaling_max` (float): Maximum gamma scaling value. Default: `0.8`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

#### Methods

##### `forward(input, **kwargs)`

Applies intensity augmentation (noise and gamma scaling) using Voxynth library.

**Parameters:**
- `input` (dict): Input dictionary with `image` (required)

**Returns:**
- `dict`: Output dictionary with augmented image

**Behavior:**
- Calls `voxynth.augment.image_augment()` with intensity augmentation parameters
- Applies noise injection and gamma scaling based on probabilities
- Returns augmented image with unchanged label, prior, and geometry

**Note:** Unlike `AugmentBase.IntensityAugmentation`, this version does not include normalization or clipping. It focuses on noise and gamma scaling only.

---

### `fsdeepnet.augmentation.augmentvoxynth.BiasFieldCorruptionAndIntensityAugmentation`

Combined bias field corruption and intensity augmentation in a single Voxynth call. More efficient than applying them separately.

#### Constructor

```python
BiasFieldCorruptionAndIntensityAugmentation(hp=None, device=None, sampling_hp=True, verbose=False)
```

**Parameters:**
- `hp` (dict, optional): Hyperparameters dictionary. Default: `{}`
  
  **Intensity Augmentation Parameters:**
  - `added_noise_probability` (float): Probability of applying noise. Default: `0.5`
  - `added_noise_max_sigma` (float): Maximum standard deviation for Gaussian noise. Default: `0.05`
  - `gamma_scaling_probability` (float): Probability of applying gamma scaling. Default: `0.5`
  - `gamma_scaling_max` (float): Maximum gamma scaling value. Default: `0.8`
  
  **Bias Field Parameters:**
  - `bias_field_probability` (float): Probability of applying bias field. Default: `0.5`
  - `bias_field_max_magnitude` (float): Maximum magnitude for bias field. Default: `0.1`
  - `bias_field_smoothing_range` (list, optional): Smoothing range `[min, max]`. If `None`, calculated from `bias_field_scale`. Default: `None`
  - `bias_field_scale` (float): Scale ratio for bias field. Default: `0.025`
  - `bias_field_generation_method` (str): Generation method - `"blur"` or `"upsample"`. Default: `"blur"`
- `device` (torch.device, optional): Device for operations. Default: `None` (auto-select)
- `sampling_hp` (bool): Whether to sample hyperparameters. Default: `True`
- `verbose` (bool): Enable verbose logging. Default: `False`

**Raises:**
- `AssertionError`: If `bias_field_generation_method` is not `"blur"` or `"upsample"`

**Behavior:**
- Calculates `bias_field_smoothing_range` from `bias_field_scale` if not provided (same logic as `BiasFieldCorruption`)
- Combines both augmentations in a single `voxynth.augment.image_augment()` call
- Applies normalization (`normalize=True`) automatically
- More efficient than applying `BiasFieldCorruption` and `IntensityAugmentation` separately

#### Methods

##### `forward(input, **kwargs)`

Applies both bias field corruption and intensity augmentation in a single call.

**Parameters:**
- `input` (dict): Input dictionary with `image`, `geom` (required)

**Returns:**
- `dict`: Output dictionary with augmented image

**Behavior:**
- Calls `voxynth.augment.image_augment()` with both bias field and intensity parameters
- Applies normalization automatically
- Copies `geom.voxsize` to make it writable
- Returns augmented image with unchanged label, prior, and geometry

**Example:**
```python
hp = {
    "biasfieldcorruptionandintensityaugmentation": {
        "bias_field_probability": 0.5,
        "bias_field_max_magnitude": 0.1,
        "bias_field_scale": 0.025,
        "bias_field_generation_method": "blur",
        "added_noise_probability": 0.5,
        "added_noise_max_sigma": 0.05,
        "gamma_scaling_probability": 0.5,
        "gamma_scaling_max": 0.8
    }
}
transforms = ["biasfieldcorruptionandintensityaugmentation"]
augment_obj = AugmentVoxynth(hp=hp, transforms=transforms)
```

---

### Differences from `AugmentBase`

**Voxynth-based augmentations:**
- Use `voxynth.augment.image_augment()` function instead of custom implementations
- May have different default hyperparameter values
- `BiasFieldCorruption` supports both `"blur"` and `"upsample"` generation methods
- `IntensityAugmentation` does not include normalization/clipping (only noise and gamma)
- `BiasFieldCorruptionAndIntensityAugmentation` combines both in a single efficient call

**When to use `AugmentVoxynth`:**
- When you want to use Voxynth's bias field generation methods
- When you need the combined bias field + intensity augmentation
- When you prefer Voxynth's implementation over the base class versions

---

## Metrics

### `fsdeepnet.metrics.Dice`

PyTorch module for calculating Dice score or Dice loss for binary or multi-class segmentation. Supports both soft Dice (on probability maps) and hard Dice (on discrete labels).

#### Constructor

```python
Dice(
    num_classes=None,
    dice_type="soft",
    smooth=1e-6,
    return_loss=True,
    dice_squared_form=False,
    **kwargs
)
```

**Parameters:**
- `num_classes` (int, optional): Number of segmentation classes (including background). Required if `dice_type="hard"`
- `dice_type` (str): Type of Dice calculation - `"soft"` (on probability maps) or `"hard"` (on discrete labels). Default: `"soft"`
- `smooth` (float): Smoothing factor to prevent division by zero. Default: `1e-6`
- `return_loss` (bool): If `True`, returns Dice loss (1 - Dice). If `False`, returns Dice score. Default: `True`
- `dice_squared_form` (bool): Whether to use squared form in union calculation. Default: `False`
- `**kwargs`: Additional keyword arguments (ignored)

**Raises:**
- `ValueError`: If `dice_type` is not `"soft"` or `"hard"`
- `AssertionError`: If `dice_type="hard"` and `num_classes` is `None`

#### Methods

##### `forward(outputs, targets, **kwargs)`

Calculates the Dice score or Dice loss.

**Parameters:**
- `outputs` (torch.Tensor): Predicted probabilities map with shape `[N, num_classes, H, W(, D)]`
- `targets` (torch.Tensor): Ground truth one-hot encoded labels with shape `[N, num_classes, H, W(, D)]`
- `**kwargs`: Additional keyword arguments (ignored)

**Returns:**
- `torch.Tensor`: 
  - If `return_loss=True`: Scalar tensor representing average Dice loss (1 - mean Dice score)
  - If `return_loss=False`: Tensor with shape `[N, num_classes]` containing Dice scores for each class

**Behavior:**
- If `dice_type="hard"`, converts probability map to one-hot encoded labels using `argmax`
- Calculates Dice score for each class using `_dice_score()`
- If `return_loss=True`, returns `1 - mean(dice_scores)`
- If `return_loss=False`, returns individual Dice scores for each class

**Example:**
```python
dice_loss = Dice(num_classes=5, dice_type="soft", return_loss=True)
loss = dice_loss(outputs, targets)  # Returns scalar loss

dice_score = Dice(num_classes=5, dice_type="hard", return_loss=False)
scores = dice_score(outputs, targets)  # Returns [N, num_classes] tensor
```

#### Private Methods

##### `_dice_score(outputs, targets)`

Calculates the Dice score between outputs and targets for each class.

**Parameters:**
- `outputs` (torch.Tensor): One-hot encoded predictions `[N, C, H, W(, D)]`
- `targets` (torch.Tensor): One-hot encoded ground truth `[N, C, H, W(, D)]`

**Returns:**
- `torch.Tensor`: Dice scores for each class with shape `[N, num_classes]`

**Formula:**
- Standard form: `Dice = (2 * intersection + smooth) / (union + smooth)` where `union = sum(outputs) + sum(targets)`
- Squared form: `Dice = (2 * intersection + smooth) / (sum(outputs²) + sum(targets²) + smooth)`

##### `_dice_loss(outputs, targets)`

Calculates the Dice loss (1 - Dice Score).

**Parameters:**
- `outputs` (torch.Tensor): One-hot encoded predictions
- `targets` (torch.Tensor): One-hot encoded ground truth

**Returns:**
- `torch.Tensor`: Dice loss (1 - Dice score)

---

### `fsdeepnet.metrics.DiceLoss`

Convenience wrapper class for Dice loss. Inherits from `Dice` and automatically sets `return_loss=True`.

#### Constructor

```python
DiceLoss(*args, **kwargs)
```

**Parameters:**
- `*args`: Positional arguments passed to `Dice` constructor
- `**kwargs`: Keyword arguments passed to `Dice` constructor (except `return_loss` which is forced to `True`)

**Behavior:**
- Automatically sets `return_loss=True` when calling parent `Dice` constructor
- All other parameters are the same as `Dice`

**Example:**
```python
dice_loss = DiceLoss(num_classes=5, dice_type="soft")
loss = dice_loss(outputs, targets)  # Returns scalar loss
```

---

### `fsdeepnet.metrics.DiceScore`

Convenience wrapper class for Dice score. Inherits from `Dice` and automatically sets `return_loss=False`.

#### Constructor

```python
DiceScore(*args, **kwargs)
```

**Parameters:**
- `*args`: Positional arguments passed to `Dice` constructor
- `**kwargs`: Keyword arguments passed to `Dice` constructor (except `return_loss` which is forced to `False`)

**Behavior:**
- Automatically sets `return_loss=False` when calling parent `Dice` constructor
- All other parameters are the same as `Dice`

**Example:**
```python
dice_score = DiceScore(num_classes=5, dice_type="hard")
scores = dice_score(outputs, targets)  # Returns [N, num_classes] tensor of scores
```

---

### `fsdeepnet.metrics.WeightedL2Loss`

PyTorch module for calculating weighted L2 loss. Used for pre-training models before Dice loss training. The loss is computed on the layer before softmax activation.

#### Constructor

```python
WeightedL2Loss(
    gt_target_value=15,
    epsilon=1e-4,
    **kwargs
)
```

**Parameters:**
- `gt_target_value` (float): Target value for the layer before softmax. When ground truth label is 1, target is `gt_target_value`. When label is 0, target is `-gt_target_value`. Default: `15`
- `epsilon` (float): Small constant added to weights to prevent division issues. Default: `1e-4`
- `**kwargs`: Additional keyword arguments (ignored)

#### Methods

##### `forward(y_pred, y_true, **kwargs)`

Calculates weighted L2 loss.

**Parameters:**
- `y_pred` (torch.Tensor): Predicted values from the layer before softmax with shape `[N, C, H, W(, D)]`
- `y_true` (torch.Tensor): Ground truth one-hot encoded labels with shape `[N, C, H, W(, D)]`
- `**kwargs`: Additional keyword arguments (ignored)

**Returns:**
- `torch.Tensor`: Scalar weighted L2 loss

**Formula:**
- Weights: `weights = 1 - y_true[:, 0] + epsilon` (higher weight for non-background pixels)
- Target values: `target = gt_target_value * (y_true * 2 - 1)` (maps one-hot to ±gt_target_value)
- Loss: `loss = sum(weights * (y_pred - target)²) / (sum(weights) * num_classes)`

**Example:**
```python
wl2_loss = WeightedL2Loss(gt_target_value=15, epsilon=1e-4)
loss = wl2_loss(penultimate_layer_output, onehot_labels)
```

---

### `fsdeepnet.metrics.WeightedCrossEntropyLoss`

PyTorch module for calculating weighted cross-entropy loss.

#### Constructor

```python
WeightedCrossEntropyLoss(weights=None)
```

**Parameters:**
- `weights` (torch.Tensor, optional): Class weights tensor. If `None`, uses uniform weights. Shape should be `[num_classes]`

#### Methods

##### `forward(y_pred, y_true)`

Calculates weighted cross-entropy loss.

**Parameters:**
- `y_pred` (torch.Tensor): Predicted logits with shape `[N, C, H, W(, D)]` or `[N, C]`
- `y_true` (torch.Tensor): Ground truth class indices (not one-hot) with shape `[N, H, W(, D)]` or `[N]`

**Returns:**
- `torch.Tensor`: Scalar weighted cross-entropy loss

**Behavior:**
- Automatically moves `weights` to the same device as `y_pred`
- Uses `torch.nn.functional.cross_entropy()` with provided weights

**Example:**
```python
class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0])  # Lower weight for background
wce_loss = WeightedCrossEntropyLoss(weights=class_weights)
loss = wce_loss(logits, class_indices)
```

---

### Functions

##### `fsdeepnet.metrics.iou_score(outputs, targets, threshold=0.5, smooth=1e-6, exclude_background=True)`

Calculates the Intersection over Union (IoU) score for non-background predictions and targets.

**Parameters:**
- `outputs` (torch.Tensor): Raw output from the model (probabilities) with shape `[N, H, W(, D)]` or `[N, C, H, W(, D)]`
- `targets` (torch.Tensor): Ground truth labels with shape `[N, H, W(, D)]`
- `threshold` (float): Threshold to convert probability to binary output. Default: `0.5`
- `smooth` (float): Small constant to avoid division by zero. Default: `1e-6`
- `exclude_background` (bool): If `True`, excludes the background label (assumed to be 0) in calculations. Default: `True`

**Returns:**
- `torch.Tensor`: Scalar tensor representing the average IoU score for all non-background classes

**Behavior:**
- Converts probabilities to binary predictions using threshold
- If `exclude_background=True`, masks out background pixels (label 0)
- Calculates IoU as: `(intersection + smooth) / (union + smooth)`

**Example:**
```python
iou = iou_score(predictions, ground_truth, threshold=0.5, exclude_background=True)
```

---

## Filter

### `fsdeepnet.filter.Filter`

Utility class for generating Gaussian filter kernels. Provides static methods for creating 1D and multi-dimensional Gaussian kernels for image filtering operations.

#### Methods

##### `Filter.gaussian_kernel_1d(sigma, max_sigma=None, truncate=2.5, radius=None, device=None, dtype=None)`

Generates a 1D Gaussian kernel with the given standard deviation.

**Parameters:**
- `sigma` (float): Standard deviation for the Gaussian kernel
- `max_sigma` (float, optional): Maximum sigma value for radius calculation. If `None`, uses `sigma`. Default: `None`
- `truncate` (float): Truncate the filter at this many standard deviations. Default: `2.5`
- `radius` (int, optional): Radius of the Gaussian kernel. If specified, the kernel size will be `2*radius + 1`, and `truncate` is ignored. Default: `None`
- `device` (torch.device, optional): Device to create the kernel on. Default: `None` (CPU)
- `dtype` (torch.dtype, optional): Data type for the kernel. Default: `None` (float32)

**Returns:**
- `torch.Tensor`: 1D Gaussian kernel tensor with shape `[2*radius + 1]`, normalized to sum to 1

**Behavior:**
- If `radius` is `None`, calculates radius as `ceil(truncate * max_sigma) / 2`
- Generates kernel values using Gaussian PDF: `exp(-(x²) / (2*σ²))`
- Normalizes the kernel so that all values sum to 1

**Formula:**
The 1D Gaussian PDF for zero-mean is:
```
(1 / (σ * √(2π))) * exp(-(x²) / (2*σ²))
```
The constant term is cancelled out during normalization.

**Example:**
```python
# Create a 1D Gaussian kernel with sigma=1.0
# Radius calculated as: ceil(2.5 * 1.0) / 2 = ceil(2.5) / 2 = 3 / 2 = 1
# Kernel size = 2*radius + 1 = 3
kernel_1d = Filter.gaussian_kernel_1d(sigma=1.0, truncate=2.5, device=torch.device("cuda"))
# Returns: tensor with shape [3], normalized to sum to 1

# Create with explicit radius
kernel_1d = Filter.gaussian_kernel_1d(sigma=1.0, radius=3, device=torch.device("cuda"))
# Returns: tensor with shape [7] (2*3+1=7), normalized to sum to 1
```

##### `Filter.gaussian_kernel(sigma, max_sigma=None, truncate=2.5, radius=None, device=None, dtype=None, separable=False)`

Generates a multi-dimensional Gaussian kernel with given standard deviations for each axis.

**Parameters:**
- `sigma` (list): Standard deviations for the Gaussian kernel. List of floats, one for each dimension (e.g., `[σx, σy, σz]` for 3D)
- `max_sigma` (float or list, optional): Maximum sigma values for radius calculation. If `None`, uses `sigma`. If scalar, applies to all dimensions. Default: `None`
- `truncate` (float): Truncate the filter at this many standard deviations. Default: `2.5`
- `radius` (int or list, optional): Radius of the Gaussian kernel. Can be:
  - `None`: Calculated from `truncate` and `max_sigma`
  - Single `int`: Applied to all dimensions
  - `list`: One radius per dimension
  If specified, kernel size along each axis will be `2*radius + 1`, and `truncate` is ignored. Default: `None`
- `device` (torch.device, optional): Device to create the kernel on. Default: `None` (CPU)
- `dtype` (torch.dtype, optional): Data type for the kernel. Default: `None` (float32)
- `separable` (bool): Whether to generate a separable kernel (not yet implemented). Default: `False`

**Returns:**
- `torch.Tensor`: Multi-dimensional Gaussian kernel tensor with shape `[k1, k2, k3, ...]` where `ki = 2*radius[i] + 1`, normalized to sum to 1

**Raises:**
- `AssertionError`: If `sigma` and `radius` (when provided as list) have different lengths

**Behavior:**
- Supports 2D and 3D (or higher dimensional) kernels
- If `radius` is `None`, calculates radius for each dimension as `ceil(truncate * max_sigma[i]) / 2`
- If `radius` is a scalar, applies it to all dimensions
- Creates a meshgrid of indices for all dimensions
- Generates kernel values using multi-dimensional Gaussian PDF
- Normalizes the kernel so that all values sum to 1

**Formula:**
The multi-dimensional Gaussian PDF for zero-mean is:
```
constant_term * exp(-(x²/σx² + y²/σy² + z²/σz²) / 2)
```
where `constant_term = 1 / ((σx*σy*σz) * (2π)^(ndims/2))`
The constant term is cancelled out during normalization.

**Note:** 
- `separable=True` is not yet implemented (TODO)
- Handling of `sigma=0` is not yet implemented (TODO)

**Example:**
```python
# Create a 2D Gaussian kernel with different sigmas for x and y
kernel_2d = Filter.gaussian_kernel(sigma=[1.0, 2.0], truncate=2.5, device=torch.device("cuda"))
# Returns: tensor with shape [k1, k2] where k1 and k2 depend on calculated radii

# Create a 3D Gaussian kernel with explicit radius
kernel_3d = Filter.gaussian_kernel(sigma=[1.0, 1.0, 1.0], radius=3, device=torch.device("cuda"))
# Returns: tensor with shape [7, 7, 7] (2*3+1=7 for each dimension)

# Create with uniform radius for all dimensions
kernel_3d = Filter.gaussian_kernel(sigma=[1.0, 1.5, 2.0], radius=2, device=torch.device("cuda"))
# Returns: tensor with shape [5, 5, 5] (2*2+1=5 for each dimension)
```

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

### `fsdeepnet.utils.utility`

Utility functions for data loading, logging, image processing, and other operations used throughout the Fsdeepnet pipeline.

#### Functions

##### `load_framedimage(file_path, orientation=None, device=None, ndims=3)`

Loads a medical image file and converts it to a PyTorch tensor. Supports both 2D and 3D images.

**Parameters:**
- `file_path` (str): Path to the image file (`.nii.gz`, `.nii`, `.mgz`, etc.)
- `orientation` (str, optional): Target orientation for 3D images (e.g., `"RAS"`). If `None`, keeps original orientation. Note: Reorientation is not yet implemented for 2D data. Default: `None`
- `device` (torch.device, optional): Device to load data on. If `None`, automatically selects CUDA if available, else CPU
- `ndims` (int): Number of dimensions (2 or 3). Default: `3`

**Returns:**
- `tuple`: Tuple containing:
  - `framedimage` (surfa.Volume): Loaded framed image object
  - `framedimage_tensor` (torch.Tensor): PyTorch tensor with shape `[C, H, W(, D)]` (non-batched)
  - `orig_orientation` (str): Original orientation of the image

**Raises:**
- `AssertionError`: If `ndims` is not 2 or 3

**Behavior:**
- Uses `surfa.load_volume()` to load both 2D and 3D images
- For 3D images, reorients to specified orientation if provided
- Converts framed data to PyTorch tensor with channels first
- Moves tensor to specified device

**Example:**
```python
framedimage, tensor, orig_ori = load_framedimage(
    "data/image.mgz", 
    orientation="RAS", 
    device=torch.device("cuda"),
    ndims=3
)
# tensor shape: [C, H, W, D] on GPU
```

##### `save_framedimage(framedimage_tensor, output_file, original_framedimage=None, geom=None, orientation=None, labels=None, onehotencoded=False, dtype=None, resample=False, method='nearest')`

Saves a PyTorch tensor as a medical image file. Supports various output formats and options.

**Parameters:**
- `framedimage_tensor` (torch.Tensor): Image tensor with shape `[C, H, W(, D)]` (non-batched)
- `output_file` (str): Path to save the output file
- `original_framedimage` (surfa.Volume, optional): Original framed image to copy geometry and metadata from
- `geom` (surfa.ImageGeometry, optional): Geometry to use for output. If `None` and `original_framedimage` is provided, uses its geometry
- `orientation` (str, optional): Target orientation for 3D images. If `None` and `original_framedimage` is provided, uses its orientation. Default: `"RAS"` if no original provided
- `labels` (surfa.LabelLookup, optional): Label lookup table for visualization
- `onehotencoded` (bool): Whether the tensor is one-hot encoded. If `True` and `ndims=2`, outputs as `surfa.Slice`. Default: `False`
- `dtype` (numpy.dtype, optional): Data type for output. If `None` and `original_framedimage` is provided, uses its dtype
- `resample` (bool): Whether to resample to original geometry if `original_framedimage` is provided. Default: `False`
- `method` (str): Resampling method - `"nearest"` or `"linear"`. Default: `"nearest"`

**Behavior:**
- Converts tensor from `[C, H, W(, D)]` to `[H, W(, D), C]` format
- Creates `surfa.Volume` or `surfa.Slice` object with specified geometry
- If `resample=True`, resamples to match `original_framedimage` geometry
- Reorients 3D images to specified orientation
- Saves to file using surfa's save functionality

**Example:**
```python
save_framedimage(
    segmentation_tensor,  # [C, H, W, D]
    "output/segmentation.mgz",
    original_framedimage=original_image,
    labels=label_lookup,
    resample=True,
    method='nearest'
)
```

##### `remap_labels(labels, mapping, return_counts=False)`

Remaps label values in a tensor according to a mapping dictionary.

**Parameters:**
- `labels` (torch.Tensor): Tensor containing label values
- `mapping` (dict): Dictionary mapping old label values to new label values: `{old_label: new_label}`
- `return_counts` (bool): Whether to return voxel counts for each non-background label. Default: `False`

**Returns:**
- `torch.Tensor` or `tuple`: 
  - If `return_counts=False`: Remapped labels tensor
  - If `return_counts=True`: Tuple of `(remapped_labels, vox_counts)` where `vox_counts` is a list of counts

**Behavior:**
- Creates a new tensor with the same shape as input
- Applies mapping to each label value
- If `return_counts=True`, counts voxels for each non-background label (label != 0)

**Example:**
```python
labels = torch.tensor([0, 1, 2, 3, 1, 2])
mapping = {1: 10, 2: 20, 3: 30}
remapped, counts = remap_labels(labels, mapping, return_counts=True)
# remapped: [0, 10, 20, 30, 10, 20]
# counts: [2, 2, 1]  # counts for labels 10, 20, 30
```

##### `onehot(labels, num_classes, device=None)`

One-hot encodes a tensor of integer labels.

**Parameters:**
- `labels` (torch.Tensor): Tensor of integer labels with shape `[N, (1,) H, W(, D)]`. If shape is `[N, 1, H, W(, D)]`, the channel dimension is squeezed
- `num_classes` (int): Number of classes (including background)
- `device` (torch.device, optional): Device for output tensor. If `None`, uses the same device as `labels`

**Returns:**
- `torch.Tensor`: One-hot encoded tensor with shape `[N, num_classes, H, W(, D)]`

**Raises:**
- `ValueError`: If output tensor is not 4D or 5D

**Behavior:**
- Uses `torch.eye()` to create one-hot encoding
- Permutes dimensions to put channels first
- Supports both 2D and 3D label maps

**Example:**
```python
labels = torch.tensor([[[0, 1, 2], [1, 2, 0]]])  # [1, 2, 3]
onehot_labels = onehot(labels, num_classes=3, device=torch.device("cuda"))
# Returns: [1, 3, 2, 3] tensor on GPU
```

##### `bbox(image, labels, verbose=False)`

Calculates the bounding box for specified labels in an image.

**Parameters:**
- `image` (torch.Tensor): Input image tensor with labels
- `labels` (list): List of label values to compute bounding box for
- `verbose` (bool): Whether to log bounding box coordinates. Default: `False`

**Returns:**
- `tuple`: Tuple containing:
  - `lowerbound` (torch.Tensor): Lower bound coordinates for each dimension `[ndims]`
  - `upperbound` (torch.Tensor): Upper bound coordinates for each dimension `[ndims]`

**Behavior:**
- Creates a binary mask for all specified labels
- Finds minimum and maximum coordinates where mask is 1
- Adds padding of 1 voxel on each side
- Returns bounds as integer tensors

**Example:**
```python
image = torch.tensor([[[0, 0, 0], [0, 1, 1], [0, 1, 1]]])
lower, upper = bbox(image, labels=[1], verbose=True)
# lower: [1, 1], upper: [2, 2] (with padding: [0, 0] to [3, 3])
```

##### `centroid(label, verbose=False)`

Calculates the centroid (center point) for non-zero labels in an image.

**Parameters:**
- `label` (torch.Tensor): Input label tensor
- `verbose` (bool): Whether to log centroid coordinates. Default: `False`

**Returns:**
- `torch.Tensor`: Centroid coordinates as integer tensor with shape `[ndims]`

**Behavior:**
- Finds bounding box of non-zero labels
- Computes center as `(lowerbound + upperbound) / 2`
- Returns integer coordinates

**Example:**
```python
label = torch.tensor([[0, 0, 0], [0, 1, 1], [0, 1, 1]])
centroid_coords = centroid(label, verbose=True)
# Returns: [1, 1] (center of bounding box)
```

##### `DataGenerator(dataloader, device=None)`

Generator function that yields batches from a DataLoader, moving data to the specified device.

**Parameters:**
- `dataloader` (torch.utils.data.DataLoader): DataLoader to iterate over
- `device` (torch.device, optional): Device to move data to. If `None`, automatically selects CUDA if available, else CPU

**Yields:**
- `tuple`: Tuple containing:
  - `n_batch` (int): Batch index
  - `images` (torch.Tensor): Image batch tensor (moved to device, float)
  - `labels` (torch.Tensor): Label batch tensor (moved to device, int)
  - `priors` (torch.Tensor): Prior batch tensor (moved to device, float)
  - `dataset_indices` (list or torch.Tensor): Indices of dataset entries in the batch

**Behavior:**
- Iterates infinitely through the DataLoader
- Moves all tensors to specified device
- Converts images and priors to float, labels to int
- Yields batches in a loop

**Example:**
```python
generator = DataGenerator(train_loader, device=torch.device("cuda"))
for n_batch, images, labels, priors, indices in generator:
    # Process batch
    if n_batch >= 1000:
        break
```

##### `set_deterministic_training(seed=42)`

Sets random seeds and enables deterministic algorithms for reproducible training.

**Parameters:**
- `seed` (int): Random seed value. Default: `42`

**Behavior:**
- Sets random seed for Python's `random` module
- Sets random seed for NumPy
- Sets random seed for PyTorch (CPU)
- If CUDA is available:
  - Sets random seed for all CUDA devices
  - Sets `torch.backends.cudnn.deterministic = True`
  - Sets `torch.backends.cudnn.benchmark = False`
- Enables deterministic algorithms with warnings only
- Sets `CUBLAS_WORKSPACE_CONFIG` environment variable for CUDA reproducibility

**Note:** For multi-process DataLoader, use `worker_init_fn()` and generator to preserve reproducibility (not yet implemented).

**Example:**
```python
set_deterministic_training(seed=42)
# All random operations will be reproducible
```

##### `print_vm_peak()`

Returns the peak virtual memory usage of the current process. Only available on Linux platforms.

**Returns:**
- `str` or `None`: Peak virtual memory as string (e.g., `"12345 kB"`), or `None` if not on Linux

**Behavior:**
- Reads `/proc/{pid}/status` file
- Extracts `VmPeak` value
- Returns formatted string with value and unit

**Example:**
```python
vm_peak = print_vm_peak()
# Returns: "12345678 kB" on Linux, None on other platforms
```

##### `gpu_report(gpu_index)`

Reports GPU information using nvidia-smi for the specified GPU.

**Parameters:**
- `gpu_index` (int): Index of GPU to report

**Behavior:**
- Runs `nvidia-smi` command to query GPU information
- Extracts: index, name, utilization, memory used, memory total, temperature
- Logs the information

**Example:**
```python
gpu_report(0)  # Report information for GPU 0
# Logs: "GPU 0: NVIDIA GeForce RTX 3090  - Utilization: 45%  - Memory Usage: 10240 / 24576 MB  - Temperature: 65C"
```

##### `config_logger(logfile=None, mode='a', level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")`

Configures the Python logging module.

**Parameters:**
- `logfile` (str, optional): Path to log file. If `None`, logs to stderr only. Default: `None`
- `mode` (str): File mode for log file (see Python file modes). Default: `'a'` (append)
- `level` (int): Logging level (e.g., `logging.DEBUG`, `logging.INFO`). Default: `logging.DEBUG`
- `format` (str): Log message format string. Default: `"%(asctime)s [%(levelname)s] %(message)s"`

**Behavior:**
- If `logfile` is provided:
  - Creates log directory if it doesn't exist
  - Configures file logging with specified mode and level
- If `logfile` is `None`:
  - Configures console handler for stderr
  - Uses a simpler format for console output (defined by `consolefmt` variable, typically `"[%(levelname)s] %(message)s"`)

**Example:**
```python
config_logger(logfile="output/training.log", level=logging.INFO)
# Logs will be written to file and console
```

##### `get_class(qualified_class_name)`

Retrieves a Python class from a fully qualified class name.

**Parameters:**
- `qualified_class_name` (str): Fully qualified class name (e.g., `"fsdeepnet.models.unet.UNet"`)

**Returns:**
- `class`: The Python class object

**Raises:**
- `AssertionError`: If module cannot be found or class doesn't exist in module

**Behavior:**
- Splits qualified name into module and class name
- Imports the module dynamically
- Retrieves the class from the module

**Example:**
```python
UNetClass = get_class("fsdeepnet.models.unet.UNet")
model = UNetClass(model_arch_dict)
```

##### `remove_duplicates(inlist, lowercase=True)`

Removes duplicates from a list while preserving order.

**Parameters:**
- `inlist` (list): Input list (can contain any type)
- `lowercase` (bool): Whether to convert strings to lowercase for comparison. Default: `True`

**Returns:**
- `list` or `None`: List with duplicates removed (in lowercase if `lowercase=True`), or `None` if input is `None`

**Behavior:**
- Converts all items to lowercase if `lowercase=True`
- Preserves first occurrence of each unique item
- Maintains original order

**Example:**
```python
aug_list = ["spatialdeformation", "intensityaugmentation", "SpatialDeformation"]
unique = remove_duplicates(aug_list)
# Returns: ["spatialdeformation", "intensityaugmentation"]
```

##### `unique_unsorted(arr)`

Gets unique elements from an array while preserving their original order.

**Parameters:**
- `arr` (numpy.ndarray): Input array

**Returns:**
- `numpy.ndarray`: Array of unique elements in their original order

**Behavior:**
- Uses `np.unique()` with `return_index=True` to find first appearance indices
- Sorts indices to preserve original order
- Returns unique elements in original order

**Example:**
```python
arr = np.array([3, 1, 2, 3, 1, 4, 2])
unique = unique_unsorted(arr)
# Returns: [3, 1, 2, 4] (preserves order of first appearance)
```

##### `write_csv(fcsv, data, header=[])`

Writes data to a CSV file.

**Parameters:**
- `fcsv` (str): Path to output CSV file
- `data` (list): List of rows, where each row is a list of values
- `header` (list): List of header rows (list of lists). Default: `[]`

**Behavior:**
- Writes header rows first (if provided)
- Writes data rows
- Uses Python's `csv` module

**Example:**
```python
data = [["subject1", 100.5, 200.3], ["subject2", 150.2, 250.1]]
header = [["Subject", "Volume1", "Volume2"]]
write_csv("output/volumes.csv", data, header=header)
```

##### `write_volume_stats(fstats, vox_counts, volumes, labels, etiv=None)`

Writes volumetric statistics to a text file in FreeSurfer-like format.

**Parameters:**
- `fstats` (str): Path to output statistics file
- `vox_counts` (list): List of voxel count lists (one per subject)
- `volumes` (list): List of volume lists (one per subject) in mm³
- `labels` (list): List of tuples `(label_id, label_name)` for each structure
- `etiv` (float, optional): Estimated Total Intracranial Volume. Default: `None`

**Behavior:**
- Writes header with metadata
- Writes table with columns: Index, SegId, NVoxels, Volume_mm3, StructName
- One row per structure per subject

**Example:**
```python
vox_counts = [[1000, 2000], [1100, 2100]]  # 2 subjects, 2 structures
volumes = [[1000.0, 2000.0], [1100.0, 2100.0]]
labels = [(1, "Left-Hippocampus"), (2, "Right-Hippocampus")]
write_volume_stats("output/stats.txt", vox_counts, volumes, labels, etiv=1500000.0)
```

##### `mask_volume(volume, mask)`

Masks a volume with a given mask, setting values outside the mask to zero.

**Parameters:**
- `volume` (numpy.ndarray or torch.Tensor): Volume to mask with shape `[B, C, H, W(, D)]` or `[C, H, W(, D)]`
- `mask` (numpy.ndarray or torch.Tensor): Mask with same shape as volume. Values > 0 define the mask region

**Returns:**
- `torch.Tensor`: Masked volume as tensor with shape matching input (with batch dimension added if needed)

**Raises:**
- `AssertionError`: If volume and mask have different shapes

**Behavior:**
- Squeezes batch dimension if present
- Creates binary mask (values > 0)
- Sets values outside mask to 0
- Returns as tensor with batch dimension

**Example:**
```python
volume = np.random.rand(1, 5, 64, 64, 64)
mask = np.zeros((1, 5, 64, 64, 64))
mask[0, :, 20:40, 20:40, 20:40] = 1
masked = mask_volume(volume, mask)
# Values outside [20:40, 20:40, 20:40] are set to 0
```

##### `get_ras_axes(aff, n_dims=3)`

Finds the RAS (Right-Anterior-Superior) axes corresponding to each dimension of a volume based on its affine matrix.

**Parameters:**
- `aff` (numpy.ndarray): Affine matrix. Can be shape `(n_dims, n_dims)`, `(n_dims+1, n_dims+1)`, or `(n_dims, n_dims+1)`
- `n_dims` (int): Number of dimensions (excluding channels). Default: `3`

**Returns:**
- `numpy.ndarray`: 1D array of length `n_dims` with axes corresponding to RAS orientations

**Behavior:**
- Inverts the affine matrix
- Finds axes with maximum absolute values in each column
- Ensures each dimension maps to a unique RAS axis

**Example:**
```python
affine = np.eye(4)  # Identity matrix
ras_axes = get_ras_axes(affine, n_dims=3)
# Returns: [0, 1, 2] (assuming standard orientation)
```

##### `find_closest_number_divisible_by_m(n, m, answer_type='lower')`

Returns the closest integer to `n` that is divisible by `m`.

**Parameters:**
- `n` (int): Input number
- `m` (int): Divisor
- `answer_type` (str): Type of answer - `"lower"` (only values ≤ n), `"higher"` (only values ≥ n), or `"closer"` (closest value). Default: `"lower"`

**Returns:**
- `int`: Closest number divisible by `m` according to `answer_type`

**Raises:**
- `Exception`: If `answer_type` is not `"lower"`, `"higher"`, or `"closer"`

**Example:**
```python
# Find closest number divisible by 8
find_closest_number_divisible_by_m(100, 8, 'lower')   # Returns: 96
find_closest_number_divisible_by_m(100, 8, 'higher')  # Returns: 104
find_closest_number_divisible_by_m(100, 8, 'closer')  # Returns: 96 (closer to 100)
```

##### `get_largest_connected_component(mask, structure=None)`

Gets the largest connected component from a binary mask.

**Parameters:**
- `mask` (numpy.ndarray): 2D or 3D binary mask (boolean or integer)
- `structure` (numpy.ndarray, optional): Connectivity structure array for `scipy.ndimage.label`. If `None`, uses default connectivity

**Returns:**
- `numpy.ndarray`: Binary mask containing only the largest connected component

**Behavior:**
- Uses `scipy.ndimage.label` to find connected components
- Identifies the component with the most voxels
- Returns binary mask of only that component
- If no components found, returns a copy of the input mask

**Example:**
```python
mask = np.array([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
largest = get_largest_connected_component(mask)
# Returns mask with only the largest connected region of 1s
```

### `fsdeepnet.checkpoint.Checkpoint`

Checkpoint management class for saving and loading PyTorch model checkpoints. Handles model state dictionaries, optimizer states, training metadata, and model architecture information.

#### Constructor

```python
Checkpoint(
    model_arch_dict=None,
    label_lookup=None,
    train_dataset_dict=None
)
```

**Parameters:**
- `model_arch_dict` (dict, optional): Model architecture configuration dictionary
- `label_lookup` (surfa.LabelLookup or str, optional): Label lookup table for visualization
- `train_dataset_dict` (dict, optional): Training dataset configuration dictionary

**Initialized Dictionary Structure:**
The checkpoint dictionary is initialized with the following keys:
- `model_arch_dict`: Model architecture configuration
- `train_dataset_dict`: Training dataset configuration (batch_size, segmentation_labels, label_mapping, inverse_label_mapping, crop_size, num_samples, input_shape, expected_num_channels)
- `label_lookup`: Label lookup table
- `epoch`: Training epoch number (initially `None`)
- `loss`: Training/validation loss (initially `None`)
- `dice`: Training/validation Dice score (initially `None`)
- `metric_type`: Metric type used ("wl2" or "dice", initially `None`)
- `model_state_dict`: Model state dictionary (initially `None`)
- `optimizer_state_dict`: Optimizer state dictionary (initially `None`)

#### Methods

##### `load(checkpoint, model=None, optimizer=None, device=None)`

Loads a checkpoint file from disk and optionally loads weights into a model and optimizer.

**Parameters:**
- `checkpoint` (str): Path to checkpoint file (`.pth` file)
- `model` (nn.Module, optional): PyTorch model to load weights into. If provided, calls `model.load_state_dict()`
- `optimizer` (torch.optim.Optimizer, optional): Optimizer to load state into. If provided and `optimizer_state_dict` exists in checkpoint, calls `optimizer.load_state_dict()`
- `device` (torch.device, optional): Device to load checkpoint on. If `None`, automatically selects CUDA if available, else CPU

**Behavior:**
- Loads the checkpoint dictionary using `torch.load()` with `map_location` set to the specified device
- Replaces `self._dict` with the loaded checkpoint dictionary
- If `model` is provided, loads model weights from `model_state_dict`
- If `optimizer` is provided and `optimizer_state_dict` exists, loads optimizer state

**Example:**
```python
checkpoint = Checkpoint()
checkpoint.load("checkpoints/model_epoch_10.pth", model=model, optimizer=optimizer, device=device)
```

##### `save(checkpoint, dict)`

Updates the checkpoint dictionary with provided values and saves to file.

**Parameters:**
- `checkpoint` (str): Path to save checkpoint file (`.pth` file)
- `dict` (dict): Dictionary of values to update in checkpoint. Keys should match checkpoint dictionary structure

**Behavior:**
- Updates `self._dict` with values from `dict`
- Saves the updated dictionary to file using `torch.save()`

**Example:**
```python
checkpoint_dict = {
    "epoch": 10,
    "metric_type": "dice",
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": 0.05,
    "dice": 0.92
}
checkpoint.save("checkpoints/model_epoch_10.pth", checkpoint_dict)
```

##### `update(dict)`

Updates the checkpoint dictionary with provided values without saving.

**Parameters:**
- `dict` (dict): Dictionary of key-value pairs to update in `self._dict`

**Example:**
```python
checkpoint.update({"epoch": 10, "loss": 0.05})
```

##### `strip(keys)`

Removes specified keys from the checkpoint dictionary.

**Parameters:**
- `keys` (str or list): Key name(s) to remove from checkpoint dictionary

**Behavior:**
- If `keys` is a string, converts it to a list
- Deletes each specified key from `self._dict`
- Logs each key removal

**Example:**
```python
checkpoint.strip("optimizer_state_dict")  # Remove single key
checkpoint.strip(["optimizer_state_dict", "loss"])  # Remove multiple keys
```

##### `rename(replacements)`

Renames keys in the checkpoint dictionary.

**Parameters:**
- `replacements` (str or list): Replacement specification(s) in format `"from_key:to_key"`

**Behavior:**
- If `replacements` is a string, converts it to a list
- For each replacement, renames the key from `from_key` to `to_key`
- Logs each rename operation

**Example:**
```python
checkpoint.rename("model_state:model_state_dict")  # Single rename
checkpoint.rename(["model_state:model_state_dict", "opt_state:optimizer_state_dict"])  # Multiple renames
```

##### `prefix_model_layer(prefix)`

Adds a prefix to all layer names in `model_state_dict`.

**Parameters:**
- `prefix` (str): Prefix to add to each layer name

**Behavior:**
- Iterates through all keys in `model_state_dict`
- Creates new keys with the prefix added
- Replaces old keys with new prefixed keys
- Logs each layer rename

**Example:**
```python
checkpoint.prefix_model_layer("unet3d.")  
# "conv1.weight" becomes "unet3d.conv1.weight"
```

##### `replace_model_layer(replacement)`

Replaces a substring in all layer names in `model_state_dict`.

**Parameters:**
- `replacement` (str): Replacement specification in format `"from:to"`

**Behavior:**
- Replaces all occurrences of `from` with `to` in each layer name
- Logs each layer rename

**Example:**
```python
checkpoint.replace_model_layer("unet3d:unet")
# "unet3d.conv1.weight" becomes "unet.conv1.weight"
```

##### `Checkpoint.print(dictionary, level=0, detail=False, indent=0, nkeys=30, keys=None, report_type=False)`

Static method to print checkpoint dictionary contents in a formatted way.

**Parameters:**
- `dictionary` (dict): Dictionary to print (typically `checkpoint.dict`)
- `level` (int): Recursion level for nested dictionaries (default: `0`)
- `detail` (bool): Whether to print detailed information (default: `False`)
- `indent` (int): Indentation level (default: `0`)
- `nkeys` (int): Maximum number of keys to report at each level (default: `30`)
- `keys` (str or list, optional): Specific keys to report. If `None`, reports all keys
- `report_type` (bool): Whether to report types for non-dict/Tensor/ndarray values (default: `False`)

**Output:**
- Prints dictionary structure with indentation
- For dictionaries: prints key names
- For tensors: prints dtype and shape
- For numpy arrays: prints dtype and shape
- For other types: prints values (optionally with types)

**Example:**
```python
Checkpoint.print(checkpoint.dict, detail=True, keys=["model_state_dict"], nkeys=10)
```

#### Properties

##### `dict`

Returns the checkpoint dictionary.

**Returns:**
- `dict`: The checkpoint dictionary containing all saved information

##### `epoch`

Returns the epoch number from checkpoint.

**Returns:**
- `int` or `None`: Training epoch number, or `None` if not set

##### `metric_type`

Returns the metric type from checkpoint.

**Returns:**
- `str` or `None`: Metric type ("wl2" or "dice"), or `None` if not set

##### `label_lookup`

Returns the label lookup table from checkpoint.

**Returns:**
- `surfa.LabelLookup` or `str` or `None`: Label lookup table, or `None` if not set

##### `model_arch_dict`

Returns the model architecture dictionary from checkpoint.

**Returns:**
- `dict` or `None`: Model architecture configuration, or `None` if not set

##### `train_dataset_dict`

Returns the training dataset dictionary from checkpoint.

**Returns:**
- `dict` or `None`: Training dataset configuration, or `None` if not set

##### `model_state_dict`

Returns the model state dictionary from checkpoint.

**Returns:**
- `dict` or `None`: Model state dictionary containing model weights, or `None` if not set

##### `optimizer_state_dict`

Returns the optimizer state dictionary from checkpoint.

**Returns:**
- `dict` or `None`: Optimizer state dictionary, or `None` if not set

---

## Scripts

### Command-Line Scripts

#### `fsdeepnet_train.py`

Training script.

**Usage:**
```bash
fspython scripts/fsdeepnet_train.py --config <config.yaml> [options]
```

#### `fsdeepnet_predict.py`

Prediction script.

**Usage:**
```bash
fspython scripts/fsdeepnet_predict.py --i <image> --o <output> --checkpoint <checkpoint> [options]
```

#### `fsdeepnet_evaluate.py`

Evaluation script.

**Usage:**
```bash
fspython scripts/fsdeepnet_evaluate.py --gt <ground_truth> --seg <segmentation> [options]
```

---

## Examples

### Training Example

```python
from fsdeepnet.training import Training
from fsdeepnet.config import Config
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
from fsdeepnet.prediction import Prediction

predictor = Prediction(device=torch.device("cuda"))
predictor.build_model(segmentation_checkpoint="checkpoints/best_model.pt")
predictor.predict(
    image_path="data/test_image.mgz",
    output_path="output/segmentation.mgz"
)
```

### Evaluation Example

```python
from fsdeepnet.evaluation import Evaluation
import numpy as np

labels = np.array([0, 2, 3, 4, 17, 41])
evaluator = Evaluation(labels)

dice_scores = evaluator.evaluate(
    gt_folder="data/ground_truth",
    eval_folder="output/segmentations",
    path_dice="output/dice_scores.npy"
)
```

