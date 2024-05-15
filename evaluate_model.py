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
from utils.metrics import dice_coefficient, iou_score
from voxynth.voxynth.augment import apply_center_crop

# Configure logging settings
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
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

# Load main configuration
config = load_config("configs/config.yaml")

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

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


if args.label_mapping:
    label_mapping_path = args.label_mapping
else:
    # Search for label_mapping.json in the model checkpoint directory
    model_checkpoint_dir = os.path.dirname(args.model_checkpoint)
    label_mapping_path = os.path.join(model_checkpoint_dir, "label_mapping.json")

# Load the label mapping
with open(label_mapping_path, "r") as f:
    label_mapping = json.load(f)

# Load the Trained Model
model = UNet3D(
    input_shape=(1, *test_dataset[0][0].shape[1:]),
    nb_features=nb_features,
    nb_levels=nb_levels,
    nb_labels=num_classes,
    use_skip=True,
    use_batchnorm=True,
).to(device)
model.load_state_dict(torch.load(args.model_checkpoint)["model_state_dict"])
model.eval()

# Evaluation Loop
total_dice, total_iou, num_samples = 0, 0, 0
start_time = time()

with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        labels = remap_labels(labels, label_mapping)
        labels = onehot(labels, num_classes=len(label_mapping), device=device)

        # Calculate metrics
        dice = dice_coefficient(
            outputs,
            labels,
            num_classes=len(label_mapping),
            phase="test",
            output_folder=unique_output_folder,
            exclude_background=True,
            # compute_dice_plots=False,
            save_dice_plots=False,
        )
        iou = iou_score(outputs, labels)
        total_dice += dice
        total_iou += iou.item()
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

        # logging.info(f"For {base_filename}, Dice score: {dice:.4f}")


# Output summary
# logging.info(
#     f"Average Dice: {total_dice / num_samples:.4f}, Average IoU: {total_iou / num_samples:.4f}"
# )
logging.info(f"Total evaluation time: {time() - start_time:.2f} seconds")
