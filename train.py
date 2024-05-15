import os
import torch
import torch.optim as optim
import logging
import argparse

import matplotlib

matplotlib.use("agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchinfo import summary
from models.model import UNet3D
from utils.losses import DiceLoss, WeightedL2Loss
from utils.metrics import dice_coefficient, iou_score
from utils.data_utils import onehot
from utils.data_utils import load_config, save_label_mapping, remap_labels
from utils.dataset import load_datasets

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,  # Set the log level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print log messages to the console
        # logging.FileHandler(os.path.join(output_folder, 'training.log'))  # Save log messages to a file
    ],
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_list_file", type=str, help="Path to the dataset list file")
parser.add_argument(
    "--crop_size", nargs="+", type=int, help="Crop size for training and validation"
)
parser.add_argument(
    "--expected_classes", nargs="+", type=int, help="Expected classes in the dataset"
)
parser.add_argument("--batch_size", type=int, help="Batch size for training and validation")
parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, help="Learning rate for the optimizer")
parser.add_argument(
    "--pre_train_learning_rate",
    type=float,
    help="Learning rate for the pre-training phase",
)
parser.add_argument(
    "--output_folder",
    type=str,
    default=None,
    help="Base folder for saving training outputs",
)
parser.add_argument(
    "--best_model_metric",
    type=str,
    default="loss",
    choices=["loss", "dice"],
    help="Metric for saving the best model (loss or dice)",
)
args = parser.parse_args()

# Load config file
config = load_config("configs/config.yaml")

# Update configuration with command-line arguments
config_updates = {k: v for k, v in vars(args).items() if v is not None}
config.update(config_updates)

# Access updated configuration values
expected_num_channels = config["dataset"]["expected_num_channels"]
expected_classes = config["dataset"]["expected_classes"]
batch_size = config["training"]["batch_size"]
nb_levels = config["model"]["nb_levels"]
nb_features = config["model"]["nb_features"]
learning_rate = config["training"]["learning_rate"]
pre_train_learning_rate = config["training"]["pre_train_learning_rate"]
num_epochs = config["training"]["num_epochs"]
pre_train_epochs = config["training"]["pre_train_epochs"]
ignore_indexes = config["training"].get("ignore_indexes", [])
output_folder = (
    args.output_folder
    if args.output_folder is not None
    else config.get("training", {}).get("output_folder", "output/training_outputs")
)
best_model_metric = config["training"]["best_model_metric"]
logging.info(f"best_model_metric: {best_model_metric}")

os.makedirs(output_folder, exist_ok=True)

# Specify the desired augmentations for training data
train_augmentations = [
    "flipping",
    "spatial_transform",
    "cropping",
    "blur_resample",
    "bias_field",
]
# train_augmentations = [
#     "flipping",
#     "cropping"
# ]

# Specify the desired augmentations for validation data (typically less aggressive)
validation_augmentations = [
    "flipping",
    "cropping",
]

test_augmentations = [
    "flipping",
    "cropping",
]

train_dataset, validation_dataset, _ = load_datasets(
    config, train_augmentations, validation_augmentations, test_augmentations
)

label_mapping = save_label_mapping(train_dataset.get_all_labels(), output_folder=output_folder)

# Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

sample_input, _ = train_dataset[0]
input_shape = sample_input.shape[1:]

# Verify dataset integrity
unique_classes = set()
for _, label in train_dataset:
    unique_values = torch.unique(label).tolist()
    unique_classes.update(unique_values)

num_classes = len(unique_classes)
logging.info("Dataset information:")
logging.info(f"Dataset list: {config['dataset']['dataset_list_file']}")
logging.info(f"Number of samples in training dataset: {len(train_dataset)}")
logging.info(f"Number of unique classes: {num_classes}")
logging.info(f"Unique class values: {sorted(unique_classes)}")
logging.info(f"Input shape: {input_shape}")
logging.info(f"Number of channels: {sample_input.shape[0]}")

assert (
    sorted(unique_classes) == expected_classes
), f"Expected classes {expected_classes}, but got {sorted(unique_classes)}"
assert (
    sample_input.shape[0] == expected_num_channels
), f"Expected {expected_num_channels} channels, but got {sample_input.shape[0]}"

# Create model
model = UNet3D(
    input_shape=(1, *input_shape),
    nb_features=nb_features,
    nb_levels=nb_levels,
    nb_labels=num_classes,
    use_skip=True,
    use_batchnorm=True,
).to(device)

# Define loss functions
pre_train_loss_fn = WeightedL2Loss(ignore_indexes=ignore_indexes)
main_loss_fn = DiceLoss(ignore_indexes=ignore_indexes)

# Define optimizers
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pre_train_optimizer = optim.Adam(model.parameters(), lr=pre_train_learning_rate)

train_losses = []
validation_losses = []
train_dices = []
validation_dices = []
best_validation_loss = float("inf")
best_validation_dice = 0.0

# print("Model Architecture:")
# summary(model, input_size=(1, 1, 160, 160, 160))
# print(model)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_dice = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        labels = remap_labels(labels, label_mapping)  # Remap labels
        labels = onehot(labels, num_classes=len(label_mapping), device=device)

        if epoch < pre_train_epochs:
            pre_train_optimizer.zero_grad()
            outputs = model(images)
            loss = pre_train_loss_fn(outputs, labels)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = main_loss_fn(outputs, labels)

        loss.backward()

        if epoch < pre_train_epochs:
            pre_train_optimizer.step()
        else:
            optimizer.step()

        train_loss += loss.item()
        # print(f"[DEBUG0]outputs shape: {outputs.shape}, labels shape: {labels.shape}")
        train_dice += dice_coefficient(
            outputs,
            labels,
            epoch=epoch,
            num_classes=len(label_mapping),
            batch_idx=batch_idx,
            output_folder=output_folder,
            phase="train",
            exclude_background=True,
            save_dice_plots=False,
        )

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)

    # Validation loop
    model.eval()
    validation_loss = 0.0
    validation_dice = 0.0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(validation_loader):
            images, labels = images.to(device), labels.to(device)

            labels = remap_labels(labels, label_mapping)
            labels = onehot(labels, num_classes=len(label_mapping), device=device)

            outputs = model(images)
            # logging.debug(
            #     f"[DEBUG-val] model outputs shape: {outputs.shape}, labels shape: {labels.shape}"
            # )

            if epoch < pre_train_epochs:
                loss = pre_train_loss_fn(outputs, labels)
            else:
                loss = main_loss_fn(outputs, labels)

            validation_loss += loss.item()
            validation_dice += dice_coefficient(
                outputs,
                labels,
                epoch=epoch,
                num_classes=len(label_mapping),
                batch_idx=batch_idx,
                output_folder=output_folder,
                phase="val",
                exclude_background=True,
                save_dice_plots=False,
            )

    validation_loss /= len(validation_loader)
    validation_dice /= len(validation_loader)

    logging.info(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {validation_loss:.4f}, Val Dice: {validation_dice:.4f}"
    )

    # Visualize learning curves
    train_losses.append(train_loss)
    train_dices.append(train_dice)
    validation_losses.append(validation_loss)
    validation_dices.append(validation_dice)

    if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
        # Create output directory for plots if it doesn't exist
        # plot_dir = 'output/training_plots'
        plot_dir = os.path.join(output_folder, "training_plots")
        os.makedirs(plot_dir, exist_ok=True)

        # 1. Loss Plot
        plt.figure()  # Create a new figure for the loss plot
        plt.plot(train_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        loss_plot_path = os.path.join(plot_dir, "loss_plot.png")
        plt.savefig(loss_plot_path)
        plt.close()

        # 2. Dice Coefficient Plot
        plt.figure()  # Create a new figure for the Dice plot
        # plt.plot([x.cpu().numpy() for x in train_dices], label="Training Dice")
        # plt.plot([x.cpu().numpy() for x in validation_dices], label="Validation Dice")
        # Corrected lines around line 259
        plt.plot(train_dices, label="Training Dice")
        plt.plot(validation_dices, label="Validation Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Coefficient")
        plt.legend()
        dice_plot_path = os.path.join(plot_dir, "dice_plot.png")
        plt.savefig(dice_plot_path)
        plt.close()

    if best_model_metric == "loss":
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": validation_loss,
                "dice": validation_dice,
            }
            checkpoint_path = os.path.join(output_folder, f"best_model_epoch{epoch+1}.pth")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint_dict, checkpoint_path)
    elif best_model_metric == "dice":
        if validation_dice > best_validation_dice:
            best_validation_dice = validation_dice
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": validation_loss,
                "dice": validation_dice,
            }
            checkpoint_path = os.path.join(output_folder, f"best_model_epoch{epoch+1}.pth")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint_dict, checkpoint_path)
