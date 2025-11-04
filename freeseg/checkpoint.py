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
        if ((optimizer is not None) and (self._dict['optimizer_state_dict'] is not None)):
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


    def update(self, dict):
        """ Update self._dict """
        self._dict.update(dict)


    def save(self, checkpoint, dict):
        """ Update checkpoint dict, save to file """
        
        self._dict.update(dict)
        torch.save(self._dict, checkpoint)


    @staticmethod
    def print(dictionary, level=0, detail=False, indent=0, nkeys=30, keys=None, report_type=False):
        import numpy
        
        # print checkpoint dict information
        if (level == 0):
            print("checkpoint information:")
            report_keys = keys if (keys is not None) else dictionary.keys()
        else:
            report_keys = dictionary.keys()

        indentation = "  " * indent
        indentation2 = "  " * (indent+2) if (level == 0) else indentation
        if (not detail and level != 0):
            print(f"{indentation} {dictionary.keys()}")
        else:
            num_keys = len(dictionary.keys())
            for (idx, k) in enumerate(report_keys):
                if (level == 0):  # print all the dict keys for the root level
                    print(f"{indentation} <<{k}>>")

                v = dictionary[k]
                if (isinstance(v, dict)):
                    if (level > 0):  # print keys only if the next level is a dict
                        print(f"{indentation} <<{k}>>")
                    Checkpoint.print(v, level+1, detail, indent+2, nkeys, report_type=report_type)
                elif (isinstance(v, torch.Tensor)):
                    print(f"{indentation2} {k} : (torch.Tensor) {v.dtype} {list(v.shape)}")
                elif (isinstance(v, numpy.ndarray)):
                    print(f"{indentation2} {k} : (numpy.ndarray) {v.dtype} {list(v.shape)}")
                elif (v is None or \
                      isinstance(v, int)   or isinstance(v, numpy.int32)   or isinstance(v, numpy.int64)   or \
                      isinstance(v, float) or isinstance(v, numpy.float32) or isinstance(v, numpy.float64) or \
                      isinstance(v, str) or isinstance(v, list) or isinstance(v, tuple)):
                    type_v = f"({type(v)})" if (report_type) else ""
                    print(f"{indentation2} {v} {type_v}") if (level == 0) else print(f"{indentation2} {k} : {v} {type_v}")
                    
                else:
                    print(f"{indentation2} {type(v)}") if (level == 0) else print(f"{indentation2} {k} : {type(v)}")

                # set limits to how many keys to report
                if (level != 0 and idx == nkeys-1 and idx != num_keys-1):
                    print(f"{indentation2} ... {num_keys-idx-1} more ...")
                    break;


    @property
    def dict(self):
        return self._dict
        

    @property
    def epoch(self):
        return self._dict.get('epoch', None)


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
        return self._dict.get('model_state_dict', None)

    
    @property
    def optimizer_state_dict(self):
        return self._dict.get('optimizer_state_dict', None)
