import torch


# ??? todo: this class is not in used yet
# ??? it will replace utils/train_utils.py
# ??? update Training/Prediction to use this class for load/save checkpoint
# ??? replace label_mapping with train_dataset_dict['labels_segmentation']
class Checkpoint:
    """
    This class handles model checkpoint io.
    """

    def __init__(self):
        self._dict = {
            "epoch" : None,
            "loss"  : None,
            "dice"  : None,
            "metric_type" : metric_type,                      
            "model_arch_dict" : [],
            "train_dataset_dict" : [],  # labels_segmentation, batch_size, crop_size, num_channels
            "label_lookup" : None,
            "model_state_dict" : [],
            "optimizer_state_dict": []
        }


    def load(self, checkpoint, model=None, optimizer=None, device=None):
        """ Loads a checkpoint file """

        if (device is None):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(checkpoint, map_location=device)  # Load the saved checkpoint

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


    def save(self, checkpoint, **kwargs)
        """ Update checkpoint dict, save to file """
        
        self._dict.update(kwargs)
        torch.save(self._dict, checkpoint)
        
        
        
