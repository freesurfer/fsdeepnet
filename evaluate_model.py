import os
import json
import torch
import logging
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from time import time
from utils.dataset import load_datasets
from models.model import UNet3D
from utils.data_utils import load_config, load_volume, save_volume, remap_labels
from utils.data_utils import onehot
from utils.metrics import DiceScore

# Configure logging settings
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # force device to cpu

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="configs/config.yaml", help="Path to the configuration file"
)
parser.add_argument("--batch_size", type=int, help="Batch size for evaluation")
parser.add_argument(
    "--model_checkpoint", type=str, required=True, help="Path to the model checkpoint"
)
parser.add_argument(
    "--label_mapping",
    type=str,
    default=None,
    help="Path to the label_mapping.json file. If not provided, the script will search for it in the model checkpoint directory.",
)
parser.add_argument(
    "--output_folder",
    type=str,
    required=True,
    help="Base folder for saving evaluation results",
)
args = parser.parse_args()

# Load main configuration file
config = load_config(args.config)

# Load datasets (including test data)
test_augmentations = [
    "cropping",
]
_, _, test_dataset = load_datasets(config, test_augmentations=test_augmentations)

# Create unique subfolder for this run
model_name = os.path.basename(args.model_checkpoint).replace(".pth", "")
unique_output_folder = os.path.join(args.output_folder, model_name)
os.makedirs(unique_output_folder, exist_ok=True)

batch_size = config["evaluation"]["batch_size"]
num_classes = config["model"]["num_classes"]
nb_features = config["model"]["nb_features"]
nb_levels = config["model"]["nb_levels"]
ignore_indexes = config["training"].get("ignore_indexes", [])
expected_num_channels = config["dataset"]["expected_num_channels"]


test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the label mapping
if args.label_mapping:
    label_mapping_path = args.label_mapping
else:
    # Search for label_mapping.json in the root dir of training output
    model_checkpoint_dir = os.path.dirname(args.model_checkpoint)
    label_mapping_path = os.path.join(model_checkpoint_dir, "..", "label_mapping.json")

with open(label_mapping_path, "r") as f:
    label_mapping = json.load(f)

# Load the Trained Model
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
    final_pred_activation="softmax",
).to(device)

model.load_state_dict(torch.load(args.model_checkpoint)["model_state_dict"])
model.eval()

dice_metric_hard = DiceScore(
    num_classes=len(label_mapping),
    input_type="prob",
    dice_type="hard",
    ignore_indexes=ignore_indexes,
    # return_loss=False
)


# Evaluation Loop
total_dice_scores = torch.zeros(len(label_mapping), device=device)
num_samples = 0
start_time = time()

dice_metric_hard = DiceScore(
    num_classes=len(label_mapping),
    input_type="prob",
    dice_type="hard",
    ignore_indexes=ignore_indexes,
)

with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device).float(), labels.to(device)

        outputs = model(images)
        labels = remap_labels(labels, label_mapping)
        labels = onehot(labels, num_classes=len(label_mapping), device=device)

        # Calculate metrics
        hard_dice_scores = dice_metric_hard(outputs, labels)

        # Get non-ignored label names
        label_names = list(label_mapping.keys())
        non_ignored_label_names = [
            label_names[i] for i in range(len(label_mapping)) if i not in ignore_indexes
        ]

        num_samples += 1

        predicted_segmentation = torch.argmax(outputs, dim=1).cpu()

        original_image_path = test_dataset.image_files[idx]
        original_image, _ = load_volume(original_image_path)
        base_filename = os.path.splitext(os.path.basename(original_image_path))[0]

        # Save the predicted volume using the base filename of the input image
        save_volume(
            predicted_segmentation,
            original_image,
            os.path.join(unique_output_folder, f"{base_filename}_prediction.mgz"),
        )

        logging.info(f"Sample {idx+1} (Hard Dice):")
        for label_idx, label_name in enumerate(non_ignored_label_names):
            dice_score = torch.mean(hard_dice_scores[:, label_idx]).item()
            logging.info(f" Class {label_name}: {dice_score}")

# Calculate average Dice scores for non-ignored classes
non_ignored_total_dice_scores = total_dice_scores[: len(non_ignored_label_names)]
non_ignored_avg_dice_scores = non_ignored_total_dice_scores / num_samples

logging.info("Average Dice Scores:")
for label_idx, label_name in enumerate(non_ignored_label_names):
    avg_dice_score = non_ignored_avg_dice_scores[label_idx].item()
    logging.info(f"Average Dice Score for Class {label_name}: {avg_dice_score}")

# Output summary
logging.info(f"Total evaluation time: {time() - start_time:.2f} seconds")
