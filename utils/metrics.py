import torch
import torch.nn as nn
import torch.nn.functional as F

class Dice(nn.Module):
    def __init__(
        self,
        num_classes,
        weights=None,
        input_type="prob",
        dice_type="soft",
        smooth=1e-6,
        ignore_indexes=None,
        return_loss=True,
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
        valid_input_types = ['prob', 'max_label']
        valid_dice_types = ['soft', 'hard']
        
        if input_type.lower() not in valid_input_types:
            raise ValueError(f"Invalid `input_type`: {input_type}. "
                            f"Choose from: {valid_input_types}")
        if dice_type.lower() not in valid_dice_types:
            raise ValueError(f"Invalid `dice_type`: {dice_type}. "
                            f"Choose from: {valid_dice_types}")
        if self.input_type == 'max_label' and self.dice_type != 'hard': 
            raise ValueError("Invalid combination: `input_type` 'max_label' "
                            "must be used with `dice_type` 'hard'.") 

    def _one_hot_encode(self, labels):
        """Convert integer labels to one-hot encoding."""
        one_hot = F.one_hot(labels.long(), num_classes=self.num_classes)  # Assuming labels have shape (N, D, H, W)
        return one_hot.permute(0, 4, 1, 2, 3)  # Match the 5D shape: (N, C, D, H, W)

    def _hard_max(self, tensor, dim):
        """Approximate hard max for differentiability."""
        tensor_max = tensor.max(dim=dim, keepdim=True)[0]
        eps_hot = torch.maximum(tensor - tensor_max + self.smooth, torch.tensor(0.0))
        return eps_hot / self.smooth

    def _dice_score(self, outputs, targets):
        """Calculate Dice for a single sample and class (or all)."""
        intersection = torch.sum(outputs * targets)
        union = torch.sum(outputs) + torch.sum(targets)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice
    
    def _dice_loss(self, outputs, targets):
        intersection = torch.sum(outputs * targets)
        return 1 - ((2. * intersection + self.smooth) / (torch.sum(outputs) + torch.sum(targets) + self.smooth))

    # def forward(self, outputs, targets):
    #     """Calculate Dice score or loss."""
    #     if self.input_type == "prob":
    #         if self.dice_type == "hard":
    #             outputs = self._hard_max(outputs, dim=1)
    #             targets = self._hard_max(targets, dim=1)
    #         elif self.dice_type == "soft":
    #             outputs = F.softmax(outputs, dim=1)
    #         else:
    #             raise ValueError("Invalid `dice_type`. Choose from 'soft' or 'hard'.")

    #     elif self.input_type == "max_label":
    #         if self.dice_type == "hard":
    #             outputs = self._one_hot_encode(torch.argmax(outputs, dim=1))
    #             targets = self._one_hot_encode(targets)
    #         else:
    #             raise ValueError("Invalid `dice_type` for 'max_label' input. Choose 'hard'.")
    #     else:
    #         raise ValueError("Invalid `input_type`. Choose from 'prob' or 'max_label'.")

    #     # Debug: Print shapes
    #     # print(f"Outputs shape: {outputs.shape}")
    #     # print(f"Targets shape: {targets.shape}")

    #     # Handle ignore_indexes
    #     mask = torch.ones_like(targets, dtype=torch.bool)
    #     for index in self.ignore_indexes:
    #         mask[:, index, ...] = targets[:, index, ...] != 1  # Corrected mask application

    #     outputs = outputs * mask.float()
    #     targets = targets * mask.float()

    #     # Calculate Dice scores for each class
    #     dice_scores = []
    #     for i in range(self.num_classes):
    #         if i not in self.ignore_indexes:
    #             dice_scores.append(self._dice_score(outputs[:, i], targets[:, i]))

    #     # Apply weights (if provided)
    #     if self.weights is not None:
    #         dice_scores = [w * d for w, d in zip(self.weights, dice_scores)]
        
    #     dice_loss = self._dice_loss(outputs, targets)

    #     # Calculate mean Dice score (if in training mode)
    #     if self.training:
    #         return dice_loss
        
    #     # Return a list of Dice scores during evaluation
    #     return dice_scores
    def forward(self, outputs, targets):
        """Calculate Dice score or loss."""
        if self.input_type == "prob":
            if self.dice_type == "hard":
                outputs = self._hard_max(outputs, dim=1)
                targets = self._hard_max(targets, dim=1)
            elif self.dice_type == "soft":
                outputs = F.softmax(outputs, dim=1)
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
            # Calculate Dice scores for each class
            dice_scores = []
            for i in range(self.num_classes):
                if i not in self.ignore_indexes:
                    dice_scores.append(self._dice_score(outputs[:, i], targets[:, i]))

            # Apply weights (if provided)
            if self.weights is not None:
                dice_scores = [w * d for w, d in zip(self.weights, dice_scores)]
            
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
