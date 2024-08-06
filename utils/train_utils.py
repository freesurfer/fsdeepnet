import torch


def load_checkpoint(filepath, model=None, optimizer=None, device=None):
    """Loads a checkpoint file to resume training."""
    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device)  # Load the saved checkpoint

    if (model is not None):
        model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
    if (optimizer is not None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
        
    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    metric_type = checkpoint.get('metric_type', None)
    model_arch_dict = checkpoint.get('model_arch_dict', [])
    label_lookup = checkpoint.get('label_lookup', None)
    best_validation_loss = checkpoint['loss']  # Load the best validation loss
    best_validation_dice = checkpoint['dice']  # Load the best validation dice

    return start_epoch, metric_type, model_arch_dict, label_lookup, best_validation_loss, best_validation_dice 
