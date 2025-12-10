import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from freeseg.utils import utility as utils


class Dice(nn.Module):
    """
    Calculates the Dice score or loss for binary or multi-class segmentation.

    Args:
        num_classes (int): Number of segmentation classes (including background).
        weights (list or None, optional): Class weights for handling imbalance.
                                           Defaults to None.
        dice_type (str, optional): Type of Dice calculation ('soft' or 'hard').
                                    Defaults to "soft".
        smooth (float, optional): Smoothing factor to prevent division by zero.
                                   Defaults to 1e-6.
        return_loss (bool, optional): If True, returns Dice loss (1 - Dice).
                                       If False, returns Dice score. Defaults to True.
    """

    def __init__(
        self,
        num_classes: int,
        dice_type: str = "soft",
        smooth: float = 1e-6,
        return_loss: bool = True,
        dice_squared_form: bool = False,
        **kwargs
    ):
        super(Dice, self).__init__()
        self.num_classes = num_classes
        self.dice_type = dice_type.lower()
        self.smooth = smooth
        self.return_loss = return_loss
        self.dice_squared_form = dice_squared_form
        
        # Input Validation
        valid_dice_types = ["soft", "hard"]
        if dice_type.lower() not in valid_dice_types:
            raise ValueError(
                f"Invalid `dice_type`: {dice_type}. " f"Choose from: {valid_dice_types}"
            )
    
    def _dice_score(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the Dice score between outputs and targets for each class."""

        # outputs shape [N, C, H, W(, D)]
        if (outputs.ndim == 5):
            ndims = (2, 3, 4)
        elif (outputs.ndim == 4):
            ndims = (2, 3)
        else:
            raise ValueError("Onehot encoded label is expected to be 4 or 5 dimensions")
        
        intersection = torch.sum(outputs * targets, dim=ndims)  # Sum across spatial dimensions
        if (not self.dice_squared_form):
            union = torch.sum(outputs, dim=ndims) + torch.sum(targets, dim=ndims)
        else:
            union = torch.square(outputs) + torch.square(targets)
            union = torch.sum(union, dim=ndims)

        dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)  # Calculate Dice for each class
        return dice_scores

    def _dice_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the Dice loss (1 - Dice Score)."""
        return 1 - self._dice_score(outputs, targets)

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, List[float]]:
        """
        Calculates the Dice score or loss.

        Args:
            outputs (torch.Tensor): Predicted probabilities map [N, num_classes, H, W(, D)]
            targets (torch.Tensor): Ground truth onehot encoded [N, num_classes, H, W(, D)]

        Returns:
            Union[torch.Tensor, List[float]]: Dice score(s) or Dice loss.
                                            If multi-class and not return_loss, returns a list of Dice scores for each class.
        """
        if self.dice_type == "hard":
            # convert the probabilities map to onehot encoded labels
            outputs = torch.argmax(outputs, dim=1)
            outputs = utils.onehot(outputs, num_classes=self.num_classes, device=outputs.device)  

        # Calculate Dice scores for each class
        dice_scores = self._dice_score(outputs, targets)

        if self.return_loss:
            return 1 - torch.mean(dice_scores)  # Return average loss
        else:
            return dice_scores  # Return individual Dice scores

class DiceLoss(Dice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, return_loss=True, **kwargs)  # Pass return_loss=True

class DiceScore(Dice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, return_loss=False, **kwargs)  # Pass return_loss=False


class WeightedL2Loss(nn.Module):
    def __init__(self, gt_target_value=15, epsilon=1e-4, **kwargs):
        super(WeightedL2Loss, self).__init__()
        self.gt_target_value = gt_target_value
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, **kwargs):
        """
        # this is original implementation.
        # y_pred is the posterior, y_true is one hot encoded ground truth. 
        # if ignore_indexes=[0], y_pred * mask.float() will leave only 
        #   the correspoding label probabilities, the rest are zeroed out
        # normalizer = torch.sum(weights) * y_pred.size(-1) makes loss very small
        #

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
        """

        # compute weighted l2 loss on the layer before the softmax
        num_classes = y_true.size(1)
        weights = torch.unsqueeze(1 - y_true[:, 0] + self.epsilon, dim=1)
        normaliser = (torch.sum(weights) * num_classes)
        # groud truth value for the layer before softmax: gt_target_value when gt = 1, -gt_target_value when gt = 0.
        w2l = torch.sum(weights * torch.square(y_pred - self.gt_target_value * (y_true * 2 - 1))) / normaliser

        # implemented in /space/metropolis/1/users/yh887/hthsuseg-billot/metrics_model.metrics_model()
        # weights = KL.Lambda(lambda x: K.expand_dims(1 - x[..., 0] + 1e-4))(labels_gt)
        # normaliser = KL.Lambda(lambda x: K.sum(x[0]) * K.int_shape(x[1])[-1])([weights, last_tensor])
        # last_tensor = KL.Lambda(
        #     lambda x: K.sum(x[2] * K.square(x[1] - (x[0] * 30 - 15))) / x[3],
        #     # lambda x: K.sum(x[2] * K.square(x[1] - (x[0] * 6 - 3))) / x[3],
        #     name='wl2')([labels_gt, last_tensor, weights, normaliser])

        """
        # in current SynthSeg/Hypothalamus ext/lab2im/layers.WeightedL2Loss()
        # target_value = 5
        # epsilon = 1e-8
        num_classes = y_true.size(1)        
        weights = torch.unsqueeze(1 - y_true[:, 0] + elf.epsilon, dim=1)
        w2l = torch.sum(weights * torch.square(y_pred - self.target_value * (2 * y_true - 1))) / (torch.sum(weights) * num_classes)
        """
        
        return w2l


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
