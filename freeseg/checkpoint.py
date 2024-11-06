import torch

class Checkpoint:
    """
    This class handles model checkpoint io.
    """

    def __init__(self, model_arch_dict=None, label_lookup=None, train_dataset_dict=None):
        self._dict = {
            "model_arch_dict" : model_arch_dict,
            "train_dataset_dict" : train_dataset_dict,  # batch_size, segmentation_labels, label_mapping, inverse_label_mapping, crop_size, num_samples, input_shape, num_channels
            "label_lookup" : label_lookup,       
            "epoch" : None,
            "loss"  : None,
            "dice"  : None,
            "metric_type" : None,                      
            "model_state_dict" : None,
            "optimizer_state_dict": None,
        }


    def load(self, checkpoint, model=None, optimizer=None, device=None):
        """ Loads a checkpoint file """

        if (device is None):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dict = torch.load(checkpoint, map_location=device)  # Load the saved checkpoint

        if (model is not None):
            model.load_state_dict(self._dict['model_state_dict'])  # Load model weights
        if (optimizer is not None):
            optimizer.load_state_dict(self._dict['optimizer_state_dict'])  # Load optimizer state

        """
        start_epoch = self._dict['epoch'] + 1  # Resume from the next epoch
        metric_type = self._dict.get('metric_type', None)
        best_validation_loss = self._dict['loss']  # Load the best validation loss
        best_validation_dice = self._dict['dice']  # Load the best validation dice

        model_arch_dict = self._dict.get('model_arch_dict', [])
        label_lookup = self._dict.get('label_lookup', None)
        train_dataset_dict = self._dict.get('train_dataset_dict', [])        

        return start_epoch, metric_type, model_arch_dict, label_lookup, best_validation_loss, best_validation_dice
        """


    def save(self, checkpoint, dict):
        """ Update checkpoint dict, save to file """
        
        self._dict.update(dict)
        torch.save(self._dict, checkpoint)


    @property
    def dict(self):
        return self._dict
        

    @property
    def epoch(self):
        return self._dict['epoch']


    @property
    def metric_type(self):
        return self._dict.get('metric_type', None)    

    
    @property
    def label_lookup(self):
        return self._dict.get('label_lookup', None)


    @property
    def model_arch_dict(self):
        return self._dict.get('model_arch_dict', None)


    @property
    def train_dataset_dict(self):
        return self._dict.get('train_dataset_dict', None)


    @property
    def model_state_dict(self):
        return self._dict['model_state_dict']

    
    @property
    def optimizer_state_dict(self):
        return self._dict['optimizer_state_dict']
