import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class Dice(nn.Module):
    """
    Calculates the Dice score or loss for binary or multi-class segmentation.

    Args:
        num_classes (int): Number of segmentation classes (including background).
        weights (list or None, optional): Class weights for handling imbalance.
                                           Defaults to None.
        input_type (str, optional): Type of input ('prob' or 'max_label').
                                     Defaults to "prob".
        dice_type (str, optional): Type of Dice calculation ('soft' or 'hard').
                                    Defaults to "soft".
        smooth (float, optional): Smoothing factor to prevent division by zero.
                                   Defaults to 1e-6.
        ignore_indexes (list or None, optional): List of class indices to ignore.
                                                  Defaults to None.
        return_loss (bool, optional): If True, returns Dice loss (1 - Dice).
                                       If False, returns Dice score. Defaults to True.
    """

    def __init__(
        self,
        num_classes: int,
        weights: Union[List[float], None] = None,
        input_type: str = "prob",
        dice_type: str = "soft",
        smooth: float = 1e-6,
        ignore_indexes: Union[List[int], None] = None,
        return_loss: bool = True,
    ):
        super(Dice, self).__init__()
        self.num_classes = num_classes
        self.weights = weights
        self.input_type = input_type.lower()
        self.dice_type = dice_type.lower()
        self.smooth = smooth
        self.ignore_indexes = ignore_indexes if ignore_indexes is not None else []
        self.return_loss = return_loss

        # Input Validation
        valid_input_types = ["prob", "max_label"]
        valid_dice_types = ["soft", "hard"]

        if input_type.lower() not in valid_input_types:
            raise ValueError(
                f"Invalid `input_type`: {input_type}. " f"Choose from: {valid_input_types}"
            )
        if dice_type.lower() not in valid_dice_types:
            raise ValueError(
                f"Invalid `dice_type`: {dice_type}. " f"Choose from: {valid_dice_types}"
            )
        if self.input_type == "max_label" and self.dice_type != "hard":
            raise ValueError(
                "Invalid combination: `input_type` 'max_label' "
                "must be used with `dice_type` 'hard'."
            )

    def _one_hot_encode(self, labels: torch.Tensor) -> torch.Tensor:
        """Converts integer labels to one-hot encoding."""
        one_hot = F.one_hot(labels.long(), num_classes=self.num_classes)
        return one_hot.permute(0, 4, 1, 2, 3)

    def _dice_score(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the Dice score between outputs and targets."""
        intersection = torch.sum(outputs * targets)
        union = torch.sum(outputs) + torch.sum(targets)
        return (2.0 * intersection + self.smooth) / (union + self.smooth)

    def _dice_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the Dice loss (1 - Dice Score)."""
        return 1 - self._dice_score(outputs, targets)

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Union[torch.Tensor, List[float]]:
        """
        Calculates the Dice score or loss.

        Args:
            outputs (torch.Tensor): Predicted segmentation (probabilities or one-hot encoded).
            targets (torch.Tensor): Ground truth segmentation.

        Returns:
            Union[torch.Tensor, List[float]]: Dice score(s) or Dice loss.
                                            If multi-class and not return_loss, returns a list of Dice scores for each class.
        """
        if self.input_type == "prob":
            if self.dice_type == "hard":
                # print(f"[debug] outputs.shape: {outputs.shape}")
                # print(f"[debug] targets.shape: {targets.shape}")
                # outputs = F.one_hot(torch.argmax(outputs, dim=1), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
                # targets = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
                outputs = torch.argmax(outputs, dim=1)
                outputs = F.one_hot(outputs, num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
                # print(f"[debug] outputs.shape: {outputs.shape}")
                # print(f"[debug] targets.shape: {targets.shape}")
                # print(f"[debug] targets: {targets}")
                if targets.dim() == 4:  # Targets are not one-hot encoded
                    targets = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
                elif targets.dim() >= 5:  # Targets are already one-hot encoded
                    pass  # Skip one-hot encoding
                else:
                    raise ValueError(f"Unexpected number of dimensions for targets: {targets.dim()}")
            
            elif self.dice_type == "soft":
                outputs = outputs
            else:
                raise ValueError("Invalid `dice_type`. Choose from 'soft' or 'hard'.")

        elif self.input_type == "max_label":
            if self.dice_type == "hard":
                outputs = self._one_hot_encode(torch.argmax(outputs, dim=1))
                targets = self._one_hot_encode(targets)
            else:
                raise ValueError("Invalid `dice_type` for 'max_label' input. Choose 'hard'.")
        else:
            raise ValueError("Invalid `input_type`. Choose from 'prob' or 'max_label'.")

        # Handle ignore_indexes
        mask = torch.ones_like(targets, dtype=torch.bool)
        for index in self.ignore_indexes:
            mask[:, index, ...] = targets[:, index, ...] != 1

        outputs = outputs * mask.float()
        targets = targets * mask.float()

        if self.return_loss:
            return self._dice_loss(outputs, targets)
        else:
            # Calculate Dice scores for each class (using broadcasting)
            class_indices = torch.tensor(
                [i for i in range(self.num_classes) if i not in self.ignore_indexes]
            )
            outputs = outputs[:, class_indices]
            targets = targets[:, class_indices]
            intersection = torch.sum(outputs * targets, dim=(2, 3, 4))
            union = torch.sum(outputs, dim=(2, 3, 4)) + torch.sum(targets, dim=(2, 3, 4))
            dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)

            # Apply weights if provided
            if self.weights is not None:
                dice_scores = dice_scores * torch.tensor(self.weights).to(dice_scores.device)

            return dice_scores


class WeightedL2Loss(nn.Module):
    def __init__(self, epsilon=1e-4, ignore_indexes=None):
        super(WeightedL2Loss, self).__init__()
        self.epsilon = epsilon
        self.ignore_indexes = ignore_indexes if ignore_indexes is not None else []

    def forward(self, y_pred, y_true):
        # Create a mask that sets all ignore_indexes to False, others to True
        mask = torch.ones_like(y_true, dtype=torch.bool)
        for index in self.ignore_indexes:
            mask &= y_true != index

        # Apply mask to both predictions and targets
        y_pred = y_pred * mask.float()
        y_true = y_true * mask.float()

        # Weights can be adjusted here if needed
        weights = (1 - torch.sum(y_true, dim=1, keepdim=True) + self.epsilon).squeeze()
        normalizer = torch.sum(weights) * y_pred.size(-1)

        loss = torch.sum(weights * torch.square(y_pred - y_true)) / normalizer
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        if self.weights is not None:
            self.weights = self.weights.to(y_pred.device)
        return F.cross_entropy(y_pred, y_true, weight=self.weights)


def iou_score(outputs, targets, threshold=0.5, smooth=1e-6, exclude_background=True):
    """
    Calculate the Intersection over Union (IoU) score for non-background predictions and targets.

    Parameters:
        outputs (torch.Tensor): The raw output from the model (probabilities).
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
