import torch
import torch.nn as nn
# import logging


# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.StreamHandler(),
#         # logging.FileHandler("training.log"),
#     ],
# )

# class DiceLoss(nn.Module):
#     "Does multi class and batch handling too"
#     def __init__(self, epsilon=1e-5):
#         super().__init__()
#         self.epsilon = epsilon

#     def forward(self, y_pred, y_true):
#         # Assuming y_pred is softmax output, you might need to apply a threshold or use argmax if not
#         y_true = y_true.float()
#         numerator = 2 * torch.sum(y_pred * y_true, dim=(2, 3, 4))
#         denominator = torch.sum(y_pred + y_true, dim=(2, 3, 4))
#         dice_loss = 1 - (numerator + self.epsilon) / (denominator + self.epsilon)
#         return dice_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, ignore_indexes=None):
        super().__init__()
        self.epsilon = epsilon
        self.ignore_indexes = ignore_indexes if ignore_indexes is not None else []

    def forward(self, y_pred, y_true):
        # print(f"y_pred shape: {y_pred.shape}, range: [{y_pred.min().item()}, {y_pred.max().item()}]")
        # print(f"y_true shape: {y_true.shape}, range: [{y_true.min().item()}, {y_true.max().item()}]")

        # Create a mask that sets all ignore_indexes to False, others to True
        mask = torch.ones_like(y_true, dtype=torch.bool)
        for index in self.ignore_indexes:
            mask &= y_true != index

        y_pred = y_pred * mask.float()
        y_true = y_true * mask.float()

        numerator = 2 * torch.sum(y_pred * y_true, dim=(2, 3, 4))
        denominator = torch.sum(y_pred + y_true, dim=(2, 3, 4))
        dice_loss = 1 - (numerator + self.epsilon) / (denominator + self.epsilon)

        # print(f"Dice loss: {dice_loss.mean().item()}")

        return dice_loss.mean()



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

        # Weights can be adjusted here if needed, potentially as a function of the non-ignored targets
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
        return nn.functional.cross_entropy(y_pred, y_true, weight=self.weights)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        ce_loss = nn.functional.cross_entropy(y_pred, y_true, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.7, ce_weight=0.3, ce_weights=None):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = WeightedCrossEntropyLoss(weights=ce_weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, y_pred, y_true):
        dice = self.dice_loss(y_pred, y_true)
        ce = self.ce_loss(y_pred, y_true)
        return self.dice_weight * dice + self.ce_weight * ce
