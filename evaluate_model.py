import os
import json
import torch
import logging
import argparse
import numpy as np
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
from time import time
import datetime
from utils.train_utils import load_checkpoint
from utils.dataset import load_datasets
from models.model import UNet
from utils.data_utils import load_config, load_volume, save_volume, remap_labels, onehot
from utils.metrics import DiceScore

"""
Usage: evaluate_model.py 
       --checkpoint <checkpoint>
       --test_root_folder <test_root_folder>
       --dataset_list_file <dataset_list_file>
       [--run_name <run_name>]
       [--batch_size <n>]
       [--expected_num_channels <n>]
       [--crop_size <W H D>]
       [--config <config.yaml>]
       [--label_mapping <label_mapping.json>]
       [--write_posteriors]
       [--cpu]

       * config.yaml need to have the same network parameters as training.
         If it is not given, config.yaml saved in the training root directory is used.
       * If <label_mapping.json> is not given, label_mapping.json in the training root directory is used.
"""

# Configure logging settings
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, help="Path to the configuration file"
)
parser.add_argument("--dataset_list_file", type=str, required=True, help="Path to the dataset list file")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
parser.add_argument("--expected_num_channels", type=int, default=1, help="expected_num_channels")
parser.add_argument("--crop_size", nargs="+", type=int, help="Crop size for training and validation")
parser.add_argument("--addctab", action='store_true', default=True, help="Embed colortable into seg output")
parser.add_argument("--noaddctab", action="store_true", help="Do not embed colortable into seg output")
                    
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
)
parser.add_argument(
    "--label_mapping",
    type=str,
    default=None,
    help="Path to the label_mapping.json file. If not provided, the script will search for it in the model checkpoint directory.",
)
parser.add_argument(
    "--test_root_folder",
    type=str,
    required=True,
    help="Base folder for saving test results",
)
parser.add_argument(
    "--run_name",
    type=str,
    default=None,
    help="Descriptive name for the run (used for naming TensorBoard log directories)",
)
parser.add_argument(
    "--write_posteriors",
    action='store_true',
    help="Save the label posteriors."
)
parser.add_argument(
    "--cpu",
    action='store_true',
    help="Run on CPU."
)

args = parser.parse_args()
if (args.cpu):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(args.noaddctab):
    args.addctab = False
addctab = args.addctab
write_posteriors = args.write_posteriors
if (args.run_name is not None):
    run_name = args.run_name
else:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}"

dataset_list_file = args.dataset_list_file    
batch_size = args.batch_size                         #config["evaluation"]["batch_size"]
expected_num_channels = args.expected_num_channels   #config["dataset"]["expected_num_channels"]

# ??? todo: remove dependency on config.yaml, further cleanup is needed ???
config = None    
if (args.config is None):
    # Search for config.yaml in the root dir of training output
    model_checkpoint_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(model_checkpoint_dir, "..", "config.yaml")    
else:
    config_path = args.config
    
config = load_config(config_path)
config["dataset"]["dataset_list_file"] = dataset_list_file
config["evaluation"]["batch_size"] = batch_size
config["dataset"]["expected_num_channels"] = expected_num_channels
if (args.crop_size is not None):
    config["preprocessing"]["crop_size"] = args.crop_size


# ??? todo: remove load_datasets() dependency on config ???
test_augmentations = ["cropping"]  # only cropping is needed, config["evaluation"].get("test_augmentations")
_, _, test_dataset = load_datasets(config, test_augmentations=test_augmentations, device=device)

# Create unique subfolder for this run
model_name = os.path.basename(args.checkpoint).replace(".pth", "")
unique_output_folder = os.path.join(args.test_root_folder, f"{model_name}", run_name)
os.makedirs(unique_output_folder, exist_ok=True)


test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ??? todo: remove dependency on label_mapping.json ???
if args.label_mapping:
    label_mapping_path = args.label_mapping
else:
    # Search for label_mapping.json in the root dir of training output
    model_checkpoint_dir = os.path.dirname(args.checkpoint)
    label_mapping_path = os.path.join(model_checkpoint_dir, "..", "label_mapping.json")

with open(label_mapping_path, "r") as f:
    label_mapping = json.load(f)

# Ensure keys in label_mapping are integers
label_mapping = {int(k): v for k, v in label_mapping.items()}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Load the Trained Model
_, _, model_arch_dict, label_lookup, _, _ = load_checkpoint(args.checkpoint, device=device)
if (not addctab):
    label_lookup = None

if (model_arch_dict):
    model = UNet(
        input_shape=model_arch_dict["input_shape"],
        ndims=model_arch_dict["ndims"],
        conv_size=model_arch_dict["conv_size"],
        pool_size=model_arch_dict["pool_size"],
        refine_conv=model_arch_dict["refine_conv"],
        nb_features=model_arch_dict["nb_features"],
        nb_levels=model_arch_dict["nb_levels"],
        nb_labels=model_arch_dict["nb_labels"],
        feat_mult=model_arch_dict["feat_mult"],
        nb_conv_per_level=model_arch_dict["nb_conv_per_level"],
        use_residuals=model_arch_dict["use_residuals"],
        use_batchnorm=model_arch_dict["use_batchnorm"],
        activation=model_arch_dict["activation"],
        final_pred_activation=model_arch_dict["final_pred_activation"]).to(device)
else:
    # empty model architecture dict
    logging.info(f"checkpoint {args.checkpoint} contains no model architecture info")
    logging.info(f"use {config_path} to create the model")
    model = UNet3D(
        input_shape=(expected_num_channels, *test_dataset[0][0].shape[1:]),
        ndims=config["model"]["ndims"],
        conv_size=config["model"]["conv_size"],
        pool_size=config["model"]["pool_size"],
        refine_conv=config["model"]["refine_conv"],
        nb_features=config["model"]["nb_features"],
        nb_levels=config["model"]["nb_levels"],
        nb_labels=len(label_mapping),
        feat_mult=config["model"]["feat_mult"],
        nb_conv_per_level=config["model"]["nb_conv_per_level"],
        use_residuals=config["model"]["use_residuals"],
        use_batchnorm=config["model"]["use_batchnorm"],
        activation=config["model"]["activation"],
        final_pred_activation=config["model"].get("final_pred_activation", "softmax"),
    ).to(device)

model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])
model.eval()

logging.info("Device: {}".format(device))
logging.info(f"model_checkpoint: {args.checkpoint}")
logging.info(f"unique_output_folder: {unique_output_folder}")
# save the config file and label_mapping.json
shutil.copyfile(config_path, os.path.join(unique_output_folder, "config.yaml"))
shutil.copyfile(dataset_list_file, os.path.join(unique_output_folder, "dataset_list.yaml"))
shutil.copyfile(label_mapping_path, os.path.join(unique_output_folder, "label_mapping.json"))

dice_metric_hard = DiceScore(
    num_classes=len(label_mapping),
    input_type="prob",
    dice_type="hard",
)

# Evaluation Loop
ignore_indexes = []  # ??? todo: remove it ???
non_ignored_label_names = [label for label, idx in label_mapping.items() if idx not in ignore_indexes]
total_dice_scores = torch.zeros(len(non_ignored_label_names), device=device)
num_samples = 0
start_time = time()

# initialize dice_scores (n_labels x n_samples)
n_labels  = len(non_ignored_label_names)
n_samples = len(test_loader)
dice_scores = np.zeros((n_labels, n_samples))

with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device).float(), labels.to(device)

        # Remap labels for metric calculation
        remapped_labels = remap_labels(labels, label_mapping)

        labels_onehot = onehot(remapped_labels, num_classes=len(label_mapping), device=device)

        (outputs, _) = model(images)

        # Calculate metrics
        hard_dice_scores = dice_metric_hard(outputs, labels_onehot)
        dice_scores[:, idx] = hard_dice_scores.detach().cpu().numpy()

        total_dice_scores += torch.mean(hard_dice_scores, dim=0)
        num_samples += 1

        predicted_segmentation = torch.argmax(outputs, dim=1)

        # Remap predicted labels to original values
        original_predictions = remap_labels(predicted_segmentation, inverse_label_mapping)

        original_image_path = test_dataset.image_files[idx]
        original_image, _, orig_orientation = load_volume(original_image_path, orientation='RAS', device=device)
        base_filename = os.path.splitext(os.path.basename(original_image_path))[0]

        # Label remapping check
        # ground_truth_unique_labels = torch.unique(remapped_labels).cpu().numpy()
        # predicted_unique_labels = torch.unique(predicted_segmentation).cpu().numpy()
        # assert set(ground_truth_unique_labels) == set(predicted_unique_labels), "Mismatch in label values"

        # Save the predicted volume using the base filename of the input image
        save_volume(
            original_predictions,
            original_image,
            os.path.join(unique_output_folder, f"{base_filename}_prediction.mgz"),
            orientation=orig_orientation,
            labels=label_lookup
        )
        save_volume(
            torch.squeeze(labels),
            original_image,
            os.path.join(unique_output_folder, f"{base_filename}_gt.mgz"),
            orientation=orig_orientation,
            labels=label_lookup
        )

        if (write_posteriors):
            posteriors = outputs.movedim(1, -1)
            save_volume(
                posteriors,
                original_image,
                os.path.join(unique_output_folder, f"{base_filename}_posteriors.mgz"),
                orientation=orig_orientation,
                labels=label_lookup
            )
            
        logging.info(f"Sample {idx+1} (Hard Dice):")
        for label_idx, label_name in enumerate(non_ignored_label_names):
            dice_score = torch.mean(hard_dice_scores[:, label_idx]).item()
            logging.info(f" Class {label_name}: {dice_score:.4f}")

# output dice_scores (n_labels x n_samples)
f_dice_scores = os.path.join(unique_output_folder, "dice_scores.npy")
np.save(f_dice_scores, dice_scores)

# Calculate average Dice scores for non-ignored classes
avg_dice_scores = total_dice_scores / num_samples

logging.info("Average Dice Scores:")
for label_idx, label_name in enumerate(non_ignored_label_names):
    avg_dice_score = avg_dice_scores[label_idx].item()
    logging.info(f"Average Dice Score for Class {label_name}: {avg_dice_score:.4f}")

# Output summary
logging.info(f"Total evaluation time: {time() - start_time:.2f} seconds")
