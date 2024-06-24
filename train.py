import os
import torch
import torch.optim as optim
import logging
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import hydra
from omegaconf import DictConfig, OmegaConf

from models.model import UNet3D
from utils.data_utils import onehot, save_label_mapping, remap_labels
from utils.dataset import load_datasets
from utils.metrics import WeightedL2Loss, DiceLoss, DiceScore

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig):
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)


    # Load datasets
    train_dataset, validation_dataset, _ = load_datasets(cfg)

    # Create and save label mapping
    all_labels = torch.cat([dataset[i][1] for dataset in [train_dataset, validation_dataset] for i in range(len(dataset))])
    label_mapping = save_label_mapping(all_labels, cfg)

    # Update the config with the label mapping
    cfg.dataset.label_mapping = label_mapping

    # Create Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # Create model
    model = UNet3D(
        input_shape=(cfg.dataset.expected_num_channels, *cfg.dataset.crop_size),
        ndims=cfg.model.ndims,
        nb_features=cfg.model.nb_features,
        nb_levels=cfg.model.nb_levels,
        nb_labels=len(cfg.dataset.label_mapping),
        feat_mult=cfg.model.feat_mult,
        nb_conv_per_level=cfg.model.nb_conv_per_level,
        use_residuals=cfg.model.use_residuals,
        use_batchnorm=cfg.model.use_batchnorm,
        activation=cfg.model.activation,
        final_pred_activation="softmax",
    ).to(device)

    # print Model Architecture
    # summary(model, input_size=(1, 1, 160, 160, 160))

    # Define loss functions
    pre_train_loss_fn = WeightedL2Loss(ignore_indexes=cfg.training.ignore_indexes)
    main_loss_fn = DiceLoss(
        num_classes=len(cfg.dataset.label_mapping),
        input_type="prob",
        dice_type="soft",
        ignore_indexes=cfg.training.ignore_indexes,
    )
    dice_metric_hard = DiceScore(
        num_classes=len(cfg.dataset.label_mapping),
        input_type="prob",
        dice_type="hard",
        ignore_indexes=cfg.training.ignore_indexes,
    )

    # Define optimizers
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    pre_train_optimizer = optim.Adam(model.parameters(), lr=cfg.training.pre_train_learning_rate)


    # TensorBoard writer
    writer = SummaryWriter(cfg.output_dir)

    # Training loop
    best_validation_loss = float("inf")
    best_validation_dice = 0.0

    for epoch in range(cfg.training.num_epochs):
        model.train()
        train_loss = 0.0
        train_hard_dices = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device).float(), labels.to(device)
            labels = remap_labels(labels, cfg.dataset.label_mapping)
            labels = onehot(labels, num_classes=len(cfg.dataset.expected_classes), device=device)

            if epoch < cfg.training.pre_train_epochs:
                pre_train_optimizer.zero_grad()
                outputs = model(images)
                loss = pre_train_loss_fn(outputs, labels)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = main_loss_fn(outputs, labels)

            loss.backward()

            if epoch < cfg.training.pre_train_epochs:
                pre_train_optimizer.step()
            else:
                optimizer.step()

            train_loss += loss.item()

            # Calculate hard Dice
            batch_hard_dice = dice_metric_hard(outputs, labels)
            train_hard_dices.append(batch_hard_dice.detach().cpu().numpy())

            # Write to TensorBoard
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar(
                "Train/Dice",
                torch.mean(torch.tensor(batch_hard_dice)),
                epoch * len(train_loader) + batch_idx,
            )

            # Visualization
            if batch_idx % 10 == 0:
                slice_index = random.randint(20, 50)
                num_examples_to_visualize = min(images.size(0), 6)

                image_slices = images[:num_examples_to_visualize, 0, slice_index].cpu()
                label_slices = labels[:num_examples_to_visualize, 0, slice_index].cpu()
                output_slices = outputs[:num_examples_to_visualize, 0, slice_index].cpu()

                image_grid = make_grid(image_slices, nrow=num_examples_to_visualize, normalize=True)
                label_grid = make_grid(label_slices, nrow=num_examples_to_visualize, normalize=True)
                output_grid = make_grid(output_slices, nrow=num_examples_to_visualize, normalize=True)

                writer.add_image("Train/Input Image", image_grid, epoch * len(train_loader) + batch_idx)
                writer.add_image("Train/Label", label_grid, epoch * len(train_loader) + batch_idx)
                writer.add_image("Train/Predicted Output", output_grid, epoch * len(train_loader) + batch_idx)

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        validation_loss = 0.0
        validation_hard_dices = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(validation_loader):
                images, labels = images.to(device).float(), labels.to(device)
                labels = remap_labels(labels, cfg.dataset.label_mapping)
                labels = onehot(labels, num_classes=len(cfg.dataset.expected_classes), device=device)

                outputs = model(images)

                if epoch < cfg.training.pre_train_epochs:
                    loss = pre_train_loss_fn(outputs, labels)
                else:
                    loss = main_loss_fn(outputs, labels)

                validation_loss += loss.item()

                # Calculate hard Dice
                batch_hard_dice = dice_metric_hard(outputs, labels)
                validation_hard_dices.append(batch_hard_dice.detach().cpu().numpy())

                # Write validation loss and Dice to TensorBoard
                writer.add_scalar(
                    "Validation/Loss", loss.item(), epoch * len(validation_loader) + batch_idx
                )
                writer.add_scalar(
                    "Validation/Dice",
                    torch.mean(torch.tensor(batch_hard_dice)),
                    epoch * len(validation_loader) + batch_idx,
                )

                # Visualization
                if batch_idx % 3 == 0:
                    slice_index = random.randint(20, 50)
                    num_examples_to_visualize = min(images.size(0), 6)

                    image_slices = images[:num_examples_to_visualize, 0, slice_index].cpu()
                    label_slices = labels[:num_examples_to_visualize, 0, slice_index].cpu()
                    output_slices = outputs[:num_examples_to_visualize, 0, slice_index].cpu()

                    image_grid = make_grid(image_slices, nrow=num_examples_to_visualize, normalize=True)
                    label_grid = make_grid(label_slices, nrow=num_examples_to_visualize, normalize=True)
                    output_grid = make_grid(output_slices, nrow=num_examples_to_visualize, normalize=True)

                    writer.add_image("Validation/Input Image", image_grid, epoch * len(validation_loader) + batch_idx)
                    writer.add_image("Validation/Label", label_grid, epoch * len(validation_loader) + batch_idx)
                    writer.add_image("Validation/Predicted Output", output_grid, epoch * len(validation_loader) + batch_idx)

        validation_loss /= len(validation_loader)

        train_dice_avg = np.mean(np.concatenate(train_hard_dices))
        validation_dice_avg = np.mean(np.concatenate(validation_hard_dices))

        log.info(
            f"Epoch [{epoch+1}/{cfg.training.num_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Dice Avg: {train_dice_avg:.4f}, "
            f"Val Loss: {validation_loss:.4f}, "
            f"Val Dice Avg: {validation_dice_avg:.4f}"
        )

        # Save the best model
        if cfg.training.best_model_metric == "loss":
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss,
                    "dice": validation_dice_avg,
                    "label_mapping": cfg.dataset.label_mapping
                }
                torch.save(checkpoint_dict, os.path.join(cfg.output_dir, "best_model.pth"))
        elif cfg.training.best_model_metric == "dice":
            if validation_dice_avg > best_validation_dice:
                best_validation_dice = validation_dice_avg
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss,
                    "dice": validation_dice_avg,
                    "label_mapping": cfg.dataset.label_mapping
                }
                torch.save(checkpoint_dict, os.path.join(cfg.output_dir, "best_model.pth"))

        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint_dict, os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch+1}.pth"))

    writer.close()
    log.info("Training completed.")

if __name__ == "__main__":
    train()
