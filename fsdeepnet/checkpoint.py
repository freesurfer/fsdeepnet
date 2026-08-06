import logging
import torch

class Checkpoint:
    """
    This class handles model checkpoint io.
    """

    def __init__(self, model_arch_dict=None, label_lookup=None, train_dataset_dict=None):
        self._dict = {
            "model_arch_dict" : model_arch_dict,
            "train_dataset_dict" : train_dataset_dict,  # batch_size, segmentation_labels, label_mapping, inverse_label_mapping, crop_size, num_samples, input_shape, expected_num_channels
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
        self._dict = torch.load(checkpoint, map_location=device, weights_only=False)  # Load the saved checkpoint

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


    def strip(self, keys):
        """ Strip keys from self._dict """
        keys = [keys] if (isinstance(keys, str)) else keys
        for k in (keys):
            logging.info(f"strip dict key {k}")
            del(self._dict[k])


    def rename(self, replacements):
        replacements = [replacements] if (isinstance(replacements, str)) else replacements
        new_dict = dict()
        for replacement in (replacements):
            (from_k, to_k) = replacement.split(":")
            logging.info(f"rename dict key {from_k} => {to_k}")
            new_dict[to_k] = self.dict[from_k]
            del(self.dict[from_k])

        if (new_dict):  # new_dict is not empty
            self._dict.update(new_dict)


    def prefix_model_layer(self, prefix):
        model_state_key = "model_state_dict"
        new_model_state = dict()
        old_keys = []
        for k, v in self.model_state_dict.items():
            new_k = f"{prefix}{k}"
            logging.info(f"prefix '{model_state_key}' layer {k} => {new_k}'")
            new_model_state[new_k] = v
            old_keys.append(k)

        for k in (old_keys):
            del(self.model_state_dict[k])
        self._dict['model_state_dict'].update(new_model_state)


    def replace_model_layer(self, replacement):
        (from_k, to_k) = replacement.split(":")
        
        model_state_key = "model_state_dict"
        new_model_state = dict()
        old_keys = []
        for k, v in self.model_state_dict.items():
            new_k = k.replace(from_k, to_k)
            logging.info(f"replace '{model_state_key}' layer {k} => {new_k}'")
            new_model_state[new_k] = v
            old_keys.append(k)

        for k in (old_keys):
            del(self.model_state_dict[k])
        self._dict['model_state_dict'].update(new_model_state)


    @staticmethod
    def print(dictionary, level=0, detail=False, indent=0, nkeys=30, keys=None, report_type=False):
        import numpy
        
        # print checkpoint dict information
        if (level == 0):
            logging.info("checkpoint information:")
            # 'keys' can be either str or list
            if (isinstance(keys, str)):
                keys = [keys]
            report_keys = keys if (keys is not None) else dictionary.keys()
        else:
            report_keys = dictionary.keys()

        indentation = "  " * indent
        indentation2 = "  " * (indent+2) if (level == 0) else indentation
        if (not detail and level != 0):
            logging.info(f"{indentation} {dictionary.keys()}")
        else:
            num_keys = len(dictionary.keys())
            for (idx, k) in enumerate(report_keys):
                if (level == 0):  # print all the dict keys for the root level
                    logging.info(f"{indentation} <<{k}>>")

                v = dictionary[k]
                if (isinstance(v, dict)):
                    if (level > 0):  # print keys only if the next level is a dict
                        logging.info(f"{indentation} <<{k}>>")
                    Checkpoint.print(v, level+1, detail, indent+2, nkeys, report_type=report_type)
                elif (isinstance(v, torch.Tensor)):
                    logging.info(f"{indentation2} {k} : (torch.Tensor) {v.dtype} {list(v.shape)}")
                elif (isinstance(v, numpy.ndarray)):
                    logging.info(f"{indentation2} {k} : (numpy.ndarray) {v.dtype} {list(v.shape)}")
                elif (v is None or \
                      isinstance(v, int)   or isinstance(v, numpy.int32)   or isinstance(v, numpy.int64)   or \
                      isinstance(v, float) or isinstance(v, numpy.float32) or isinstance(v, numpy.float64) or \
                      isinstance(v, str) or isinstance(v, list) or isinstance(v, tuple) or isinstance(v, set)):
                    type_v = f"({type(v)})" if (report_type) else ""
                    logging.info(f"{indentation2} {v} {type_v}") if (level == 0) else logging.info(f"{indentation2} {k} : {v} {type_v}")
                    
                else:
                    logging.info(f"{indentation2} {type(v)}") if (level == 0) else logging.info(f"{indentation2} {k} : {type(v)}")

                # set limits to how many keys to report
                if (level != 0 and idx == nkeys-1 and idx != num_keys-1):
                    logging.info(f"{indentation2} ... {num_keys-idx-1} more ...")
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

    # backward compatibility
    # older model checkpoints have model class saved under 'name' instead of 'class'
    @property
    def model_name(self):
        arch_dict = self._dict.get("model_arch_dict", {})
        the_model_class = arch_dict.get("class", None)
        if (the_model_class is None):
            the_model_class = arch_dict.get("name", None)
        
        return the_model_class
