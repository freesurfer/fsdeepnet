import os
import torch
import torch.optim as optim
import logging
import argparse
import datetime
import random
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchinfo import summary
from models.model import UNet3D
from utils.data_utils import onehot
from utils.data_utils import load_config, save_label_mapping, remap_labels
from utils.dataset import load_datasets
from utils.metrics import WeightedL2Loss, DiceLoss, DiceScore

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,  # Set the log level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to terminal
    ],
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="configs/config.yaml", help="Path to the configuration file"
)
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
    "--run_name",
    type=str,
    default=None,
    help="Descriptive name for the run (used for naming TensorBoard log directories)",
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
config = load_config(args.config)

# Update configuration with command-line arguments
config_updates = {k: v for k, v in vars(args).items() if v is not None}
config.update(config_updates)

# Access updated configuration values
ndims = config["model"]["ndims"]
conv_size = config["model"]["conv_size"]
pool_size = config["model"]["pool_size"]
use_residuals = config["model"]["use_residuals"]
refine_conv = config["model"]["refine_conv"]
nb_levels = config["model"]["nb_levels"]
nb_features = config["model"]["nb_features"]
nb_conv_per_level = config["model"]["nb_conv_per_level"]
use_batchnorm = config["model"]["use_batchnorm"]
activation = config["model"]["activation"]
feat_mult = config["model"]["feat_mult"]

expected_num_channels = config["dataset"]["expected_num_channels"]
expected_classes = config["dataset"]["expected_classes"]

batch_size = config["training"]["batch_size"]
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

# # Create TensorBoard writer
# writer = SummaryWriter(f"{output_folder}/tensorboard_logs")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
parent_output_folder = "tensorboard_logs"  # Common parent directory for all TensorBoard logs
run_name = args.run_name if args.run_name is not None else f"run_{timestamp}"
output_folder = os.path.join(parent_output_folder, run_name)  # Output folder for current run
best_model_dir = os.path.join(output_folder, "best_models")  # Folder for best models
checkpoint_dir = os.path.join(output_folder, "checkpoints")  # Folder for checkpoints
os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

writer = SummaryWriter(output_folder)

# Specify the desired augmentations for training data
train_augmentations = config["preprocessing"].get("train_augmentations")

validation_augmentations = [
    "cropping",
]

test_augmentations = [
    "cropping",
]

train_dataset, validation_dataset, _ = load_datasets(
    config, train_augmentations, validation_augmentations, test_augmentations
)

# Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

sample_input_shape, unique_classes, all_labels = train_dataset.preload()
input_shape = sample_input_shape[1:]

label_mapping = save_label_mapping(all_labels, output_folder=output_folder)

num_classes = len(unique_classes)

assert (
    sorted(unique_classes) == expected_classes
), f"Expected classes {expected_classes}, but got {sorted(unique_classes)}"
assert (
    sample_input_shape[0] == expected_num_channels
), f"Expected {expected_num_channels} channels, but got {sample_input_shape[0]}"

logging.info("Device: {}".format(device))
logging.info("Dataset information:")
logging.info(f"Dataset list: {config['dataset']['dataset_list_file']}")
logging.info(f"Number of samples in training dataset: {len(train_dataset)}")
logging.info(f"Number of unique classes: {num_classes}")
logging.info(f"Unique class values: {sorted(unique_classes)}")
logging.info(f"Input shape: {input_shape}")
logging.info(f"Number of channels: {sample_input_shape[0]}")

# Create model
model = UNet3D(
    input_shape=(expected_num_channels, *input_shape),
    ndims=ndims,
    conv_size=conv_size,
    pool_size=pool_size,
    refine_conv=refine_conv,
    nb_features=nb_features,
    nb_levels=nb_levels,
    nb_labels=num_classes,
    feat_mult=feat_mult,
    nb_conv_per_level=nb_conv_per_level,
    use_residuals=use_residuals,
    use_batchnorm=use_batchnorm,
    activation=activation,
    final_pred_activation="softmax",
).to(device)

# print Model Architecture
# summary(model, input_size=(1, 1, 160, 160, 160))

# Define loss functions
pre_train_loss_fn = WeightedL2Loss(ignore_indexes=ignore_indexes)

main_loss_fn = DiceLoss(
    num_classes=len(label_mapping),
    input_type="prob",
    dice_type="soft",
    ignore_indexes=ignore_indexes,
)

dice_metric_hard = DiceScore(
    num_classes=len(label_mapping),
    input_type="prob",
    dice_type="hard",
    ignore_indexes=ignore_indexes,
    # return_loss=False,
)

# Define optimizers
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pre_train_optimizer = optim.Adam(model.parameters(), lr=pre_train_learning_rate)

train_losses = []
validation_losses = []
train_dices = []
validation_dices = []
best_validation_loss = float("inf")
best_validation_dice = 0.0


# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_hard_dices = []
    num_train_batches = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device).float(), labels.to(device)
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

        # --- Metrics Calculation ---
        # Calculate hard Dice
        batch_hard_dice = dice_metric_hard(outputs, labels)
        train_hard_dices.append(batch_hard_dice.detach().cpu().numpy())

        num_train_batches += 1

        # Write to TensorBoard every batch
        writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar(
            "Train/Dice",
            torch.mean(torch.tensor(batch_hard_dice)),
            epoch * len(train_loader) + batch_idx,
        )

        # logging.info(
        #     f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
        #     f"Train Loss: {loss.item():.4f}, "
        ##    f"Train Soft Dice: {np.mean(train_soft_dices, axis=0)}, "
        #     f"Train Hard Dice: {np.mean(train_hard_dices, axis=0)}"
        # )

        # --- TensorBoard Visualization (Inside Training Loop) ---
        if batch_idx % 10 == 0:  # Visualize every 10 batches (adjust as needed)
            slice_index = random.randint(20, 50)  # Choose representative slice index
            num_examples_to_visualize = min(images.size(0), 6)  # visualize upto 6 examples

            # Get slices from different examples in the batch
            image_slices = images[:num_examples_to_visualize, 0, slice_index].cpu()
            label_slices = labels[:num_examples_to_visualize, 0, slice_index].cpu()
            output_slices = outputs[:num_examples_to_visualize, 0, slice_index].cpu()

            # Create grids (using a colormap if needed)
            image_grid = make_grid(image_slices, nrow=num_examples_to_visualize, cmap="viridis")
            label_grid = make_grid(label_slices, nrow=num_examples_to_visualize, cmap="viridis")
            output_grid = make_grid(output_slices, nrow=num_examples_to_visualize, cmap="viridis")

            # Add images to TensorBoard
            writer.add_image("Train/Input Image", image_grid, epoch * len(train_loader) + batch_idx)
            writer.add_image("Train/Label", label_grid, epoch * len(train_loader) + batch_idx)
            writer.add_image(
                "Train/Predicted Output", output_grid, epoch * len(train_loader) + batch_idx
            )

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation loop
    model.eval()
    validation_loss = 0.0
    validation_hard_dices = []
    num_val_batches = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(validation_loader):
            images, labels = images.to(device).float(), labels.to(device)
            labels = remap_labels(labels, label_mapping)
            labels = onehot(labels, num_classes=len(label_mapping), device=device)

            outputs = model(images)

            if epoch < pre_train_epochs:
                loss = pre_train_loss_fn(outputs, labels)
            else:
                loss = main_loss_fn(outputs, labels)

            validation_loss += loss.item()

            # --- Metrics Calculation ---
            # Calculate hard Dice
            batch_hard_dice = dice_metric_hard(outputs, labels)
            validation_hard_dices.append(batch_hard_dice.detach().cpu().numpy())

            num_val_batches += 1

            # logging.info(
            #     f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(validation_loader)}], "
            #     f"Val Loss: {loss.item():.4f}, "
            ##    f"Val Soft Dice: {np.mean(validation_soft_dices, axis=0)}, "
            #     f"Val Hard Dice: {np.mean(validation_hard_dices, axis=0)}"
            # )

            # Write validation loss and Dice to TensorBoard (once per epoch)
            writer.add_scalar(
                "Validation/Loss", validation_loss, epoch * len(validation_loader) + batch_idx
            )
            writer.add_scalar(
                "Validation/Dice",
                torch.mean(torch.tensor(batch_hard_dice)),
                epoch * len(validation_loader) + batch_idx,
            )

            # --- TensorBoard Visualization (Inside Validation Loop) ---
            if batch_idx % 3 == 0:  # Visualize every 3 batches (adjust as needed)
                slice_index = random.randint(20, 50)

                # Get slices from different examples in the batch
                image_slices = images[:num_examples_to_visualize, 0, slice_index].cpu()
                label_slices = labels[:num_examples_to_visualize, 0, slice_index].cpu()
                output_slices = outputs[:num_examples_to_visualize, 0, slice_index].cpu()

                # Create grids (using a colormap if needed)
                image_grid = make_grid(image_slices, nrow=num_examples_to_visualize, cmap="viridis")
                label_grid = make_grid(label_slices, nrow=num_examples_to_visualize, cmap="viridis")
                output_grid = make_grid(
                    output_slices, nrow=num_examples_to_visualize, cmap="viridis"
                )

                # Add images to TensorBoard
                writer.add_image(
                    "Validation/Input Image", image_grid, epoch * len(validation_loader) + batch_idx
                )
                writer.add_image(
                    "Validation/Label", label_grid, epoch * len(validation_loader) + batch_idx
                )
                writer.add_image(
                    "Validation/Predicted Output",
                    output_grid,
                    epoch * len(validation_loader) + batch_idx,
                )

    validation_loss /= len(validation_loader)
    validation_losses.append(validation_loss)

    train_dice_avg = torch.mean(torch.tensor(np.concatenate(train_hard_dices)))
    validation_dice_avg = torch.mean(torch.tensor(np.concatenate(validation_hard_dices)))

    logging.info(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Dice Avg: {train_dice_avg:.4f}, "
        f"Val Loss: {validation_loss:.4f}, "
        f"Val Dice Avg: {validation_dice_avg:.4f}"
    )

    # Save the best model
    if best_model_metric == "loss":
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": validation_loss,
                "dice": validation_dice_avg,
            }
            checkpoint_path = os.path.join(
                best_model_dir,
                f"best_model_epoch{epoch+1}_val_loss{validation_loss:.4f}_val_dice{validation_dice_avg:.4f}.pth",
            )
            torch.save(checkpoint_dict, checkpoint_path)
    elif best_model_metric == "dice":
        if validation_dice_avg > best_validation_dice:
            best_validation_dice = validation_dice_avg
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": validation_loss,
                "dice": validation_dice_avg,
            }
            checkpoint_path = os.path.join(
                best_model_dir,
                f"best_model_epoch{epoch+1}_val_loss{validation_loss:.4f}_val_dice{validation_dice_avg:.4f}.pth",
            )
            torch.save(checkpoint_dict, checkpoint_path)

    # Save periodic checkpoints
    if (epoch + 1) % 10 == 0:  # Save every 10 epochs
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_epoch{epoch+1}_val_loss{validation_loss:.4f}_val_dice{validation_dice_avg:.4f}.pth",
        )
        torch.save(checkpoint_dict, checkpoint_path)
