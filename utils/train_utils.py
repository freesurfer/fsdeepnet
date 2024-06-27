import torch


def load_checkpoint(filepath, model, optimizer):
    """Loads a checkpoint file to resume training."""
    checkpoint = torch.load(filepath)  # Load the saved checkpoint

    model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    best_validation_loss = checkpoint['loss']  # Load the best validation loss
    best_validation_dice = checkpoint['dice']  # Load the best validation dice

    return start_epoch, best_validation_loss, best_validation_dice 