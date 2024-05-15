import os
import torch
import matplotlib.pyplot as plt

def dice_coefficient(
    outputs,
    targets,
    num_classes,
    epoch=None,
    batch_idx=None,
    output_folder=None,
    phase=None,
    threshold=0.5,
    smooth=1e-6,
    exclude_background=True,
    save_dice_plots=False,
):
    """Calculate the Dice score for the predicted outputs and ground truth targets."""
    dice_scores = []
    num_subplots = num_classes - 1 if exclude_background else num_classes

    if save_dice_plots:
        # Create a figure for visualizing results
        fig, axes = plt.subplots(num_subplots, 2, figsize=(10, 10))

        # Choose a random slice index to visualize
        slice_idx = torch.randint(0, outputs.shape[-1], (1,)).item()

    # Loop through each class and calculate Dice score
    for i, class_idx in enumerate(range(1 if exclude_background else 0, num_classes)):
        # Generate binary predictions and targets for this class
        predicted = (outputs[:, class_idx] > threshold).float()
        true = (targets[:, class_idx] > 0).float()

        # Calculate intersection and union
        intersection = torch.sum(predicted * true)
        union = torch.sum(predicted) + torch.sum(true)

        # Calculate Dice score and add to list
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice_score.item())

        if save_dice_plots:
            # Plot the output slice for the current class
            if num_subplots == 1:
                ax = axes[0]
                ax_truth = axes[1]
            else:
                ax = axes[i, 0]
                ax_truth = axes[i, 1]

            output_slice = outputs[0, class_idx, :, :, slice_idx].detach().cpu().numpy()
            ax.imshow(output_slice, cmap="gray")
            ax.set_title(f"Output Class {class_idx} - Dice: {dice_score:.3f}")

            # Plot the corresponding ground truth for this class
            true_slice = true[0, :, :, slice_idx].detach().cpu().numpy()
            ax_truth.imshow(true_slice, cmap="gray")
            ax_truth.set_title(f"Ground Truth Class {class_idx}")

    if save_dice_plots:
        # Ensure the dice_plots directory exists
        if output_folder:
            dice_plots_dir = os.path.join(output_folder, "dice_plots")
            os.makedirs(dice_plots_dir, exist_ok=True)

            # Create a unique filename based on epoch, batch index, and phase
            filename = f"dice_plot_{phase}_epoch_{epoch}_batch_{batch_idx}.png"
            filepath = os.path.join(dice_plots_dir, filename)

            # Save the plot to the specified path
            plt.savefig(filepath)
            plt.close(fig)

    # Calculate mean Dice score across all classes
    mean_dice = sum(dice_scores) / len(dice_scores)
    return mean_dice


def iou_score(outputs, targets, threshold=0.5, smooth=1e-6, exclude_background=True):
    """
    Calculate the Intersection over Union (IoU) score, also known as the Jaccard index,
    for non-background predictions and targets.

    Parameters:
        outputs (torch.Tensor): The raw output from the model, expected to be probabilities.
        targets (torch.Tensor): The ground truth labels.
        threshold (float, optional): Threshold to convert probability to binary output.
        smooth (float, optional): Small constant to avoid division by zero.
        exclude_background (bool, optional): If True, exclude the background label in calculations.

    Returns:
        torch.Tensor: The average IoU score for all non-background classes.
    """
    outputs = (outputs > threshold).float()

    # Exclude background
    if exclude_background:
        mask = targets != 0  # Assuming '0' is the background label
        outputs = outputs * mask
        targets = targets * mask

    # Flatten the tensors
    outputs_flat = outputs.view(-1)
    targets_flat = targets.contiguous().view(-1)

    intersection = (outputs_flat * targets_flat).sum()
    total = (outputs_flat + targets_flat).sum()
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou
