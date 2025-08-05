import os
import logging
import numpy as np
import torch

from freeseg import augmentation
from freeseg.utils import load_framedimage, save_framedimage, remap_labels, onehot, get_class, remove_duplicates

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, config, augment_obj, dataset_dict=None, image=None, label=None, priors=None, device=None, keep_trainset_in_memory=False):
        """
        SegmentationDataset Constructor

        dataset_dict : dict (optional)
          Input dataset dict containing input image volumes and label maps
        image : list (optional)
          Input image volume(s)
        label : list (optional)
          Input label map(s)
        """

        assert ((dataset_dict is not None) or (image is not None and label is not None)), \
            "Must provide input image/label using 'dataset_dict' or 'image/label'"

        self.num_entries = len(dataset_dict) if (dataset_dict is not None) else len(image)
        self.num_channels = config["dataset"]["expected_num_channels"]
        self.ndims = config["model"]["ndims"]
        self.expected_classes = config["dataset"]["expected_classes"]
        self.num_classes = len(sorted(config["dataset"]["expected_classes"]))
        self.label_mapping = config["dataset"]["label_mapping"]

        assert (self.ndims == 3 or self.ndims == 2), "Model supports 3D or 2D"
                
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.save_volumes = None
        self.output_dir = config["preprocessing"].get("augmentation_dir", None)
        if ((self.output_dir is not None) and (not os.path.exists(self.output_dir))):
            os.makedirs(self.output_dir)        
        if (keep_trainset_in_memory and self.output_dir is not None):
            logging.error(f"'--keep_trainset_in_memory' doesn't work with saving augmentation volumes") 
            raise ValueError("'--keep_trainset_in_memory' doesn't work with saving augmentation volumes")

        self.keep_trainset_in_memory = keep_trainset_in_memory
        self.images = []
        self.image_tensors , self.label_tensors, self.prior_tensors = [], [], []
        
        # Extract image, label, priors file paths
        self.image_files, self.label_files, self.priors_files = [], [], []
        if (dataset_dict is not None):
            for item in dataset_dict:
                self.image_files.append(item["image_filepath"])
                self.label_files.append(item["label_filepath"])
                if (item.get("prior_filepath")):
                    self.priors_files.append(item["prior_filepath"])
        elif (image is not None and label is not None):
            self.image_files = image
            self.label_files = label
            if (priors is not None):
                self.priors_files = priors

        assert (len(self.image_files) == len(self.label_files)), "image and label need to be the same length"
        if (self.haspriors()):
            assert (len(self.label_files) == len(self.priors_files)), "label and priors need to be the same length"

        self.data_augment = augment_obj
        if (self.data_augment is not None):
            augmentation.check_augmentations(self.data_augment)
        

    def haspriors(self):
        return True if (len(self.priors_files) > 0) else False

    
    def __len__(self):
        return self.num_entries


    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]

        if (not self.keep_trainset_in_memory):
            # Load image and label using the load_framedimage function
            image, image_tensor, _ = load_framedimage(image_path, orientation="RAS", device=self.device, ndims=self.ndims)
            label, label_tensor, _ = load_framedimage(label_path, orientation="RAS", device=self.device, ndims=self.ndims)
            
            # load priors if they are provided
            priors_tensor = None        
            if (self.haspriors()):
                priors_path = self.priors_files[index]        
                sfprior, priors_tensor, _  = load_framedimage(priors_path, orientation="RAS", device=self.device, ndims=self.ndims)

            if (self.output_dir is not None):
                # where/whether to save preprocessed data
                self.save_volumes = f"{index+1:04d}."+os.path.splitext(os.path.basename(image_path))[0]
        else:
            # retrieve preloaded data
            # saving augmentated volumes will not work when keep_trainset_in_memory=True
            # because the input volume names are not available
            label = None
            sfprior = None
            image = self.images[index]            
            image_tensor  = self.image_tensors[index]
            label_tensor  = self.label_tensors[index]
            priors_tensor = self.prior_tensors[index]

        # Apply data augmentation
        if (self.data_augment is not None):
            # image.geom.voxsize returned from surfa.load_volume() is (3, 1)
            # extract voxsizes to match {image_tensor.ndim-1}D data
            # make it writeable or voxynth.augment.image_augment() will complain non-writable numpy array
            voxsize = np.copy(image.geom.voxsize[:image_tensor.ndim-1])
            augmented_image_tensor, augmented_label_tensor, augmented_priors_tensor = \
                augmentation.apply_augmentations(
                    self.data_augment,
                    image_tensor,
                    label_tensor,
                    image,
                    label,
                    voxsize=voxsize,
                    priors_tensor=priors_tensor,
                    save_volumes=self.save_volumes,
                )
                
            # freeseg.utils.remap_labels() and freeseg.utils.onehot() expect batched tensor [N, 1, H, W(, D)]
            # add batch axis before calling remap_labels() and onehot()
            augmented_label_tensor = augmented_label_tensor.int().unsqueeze(0)
            onehot_augmented_label_tensor = remap_labels(augmented_label_tensor, self.label_mapping)
            onehot_augmented_label_tensor = onehot(onehot_augmented_label_tensor, num_classes=self.num_classes, device=self.device)
            # remove the added batch axis, DataLoader will batch the tensor based on batch_size
            onehot_augmented_label_tensor = onehot_augmented_label_tensor.squeeze(0)

            # ??? todo: move the logic to augmentation.__init__.py
            if (self.output_dir is not None):
                out_label_onehot = os.path.join(self.output_dir, self.save_volumes + f"_augmented_label_onehot.mgz")
                save_framedimage(onehot_augmented_label_tensor, out_label_onehot, onehotencoded=True)

            if (augmented_priors_tensor is None):
                # torch.utils.data.DataLoader can't return NoneType, make an empty tensor with 0 elements
                augmented_priors_tensor = torch.empty(0, *onehot_augmented_label_tensor.shape[1:], device=augmented_image_tensor.device)

            return index, augmented_image_tensor, onehot_augmented_label_tensor, augmented_priors_tensor
        else:
            # freeseg.utils.remap_labels() and freeseg.utils.onehot() expect batched tensor [N, 1, H, W(, D)]
            # add batch axis before calling remap_labels() and onehot()
            onehot_label_tensor = remap_labels(label_tensor.int().unsqueeze(0), self.label_mapping)
            onehot_label_tensor = onehot(onehot_label_tensor, num_classes=self.num_classes, device=self.device)
            # remove the added batch axis, DataLoader will batch the tensor based on batch_size
            onehot_label_tensor = onehot_label_tensor.squeeze(0)
            if (priors_tensor is None):
                # torch.utils.data.DataLoader can't return NoneType, make an empty tensor with 0 elements
                priors_tensor = torch.empty(0, *onehot_label_tensor.shape[1:], device=image_tensor.device)
            
            return index, image_tensor, onehot_label_tensor, priors_tensor


    def preload(self):
        """preprocesses all label maps, retrieve input tensor shape and unique classes."""

        logging.info("Perform dataset checking ...")

        expected_num_channels = self.num_channels

        label_lookup = None
        unique_classes = set()
        for n in range(self.num_entries):
            f_label, f_image = self.label_files[n], self.image_files[n]

            label, label_tensor, _ = load_framedimage(f_label, orientation="RAS", device=self.device, ndims=self.ndims)
            image, image_tensor, _ = load_framedimage(f_image, orientation="RAS", device=self.device, ndims=self.ndims)

            # label_tensor and image_tensor are non-batched [C, H, W (,D)]
            assert (self.ndims == label_tensor.ndim - 1), f"Expected {self.ndims}D label, but got {label_tensor.ndim - 1}D"            
            #assert (self.ndims == image_tensor.ndim - 1), f"Expected {self.ndims}D image, but got {image_tensor.ndim - 1}D"
            assert (label_tensor.shape == image_tensor.shape), \
                f"image and label need to be in the same shape. label {f_label} has shape {label_tensor.shape}, image {f_image} has shape {image_tensor.shape}"

            prior, prior_tensor = None, None
            if (self.haspriors()):
                f_prior = self.priors_files[n]
                prior, prior_tensor, _ = load_framedimage(f_prior, orientation="RAS", device=self.device, ndims=self.ndims)
                
                # prior_tensor is non-batched [self.num_classes, H, W (,D)]
                assert (list(prior_tensor.shape) == [self.num_classes, *label_tensor.shape[1:]]), \
                    f"Expected prior shape [self.num_classes, *label_tensor.shape[1:]], but got {list(prior_tensor.shape)}"              

            input_shape = image_tensor.shape
            assert (input_shape[0] == expected_num_channels), \
                f"Expected {expected_num_channels} channels, but got {input_shape[0]}"
            
            if (self.keep_trainset_in_memory):
                self.images.append(image)
                self.image_tensors.append(image_tensor)
                self.label_tensors.append(label_tensor)
                self.prior_tensors.append(prior_tensor)

            if (label_lookup is None):
                label_lookup = image.labels if (image.labels is not None) else label.labels

            unique_values = np.unique(label.data).astype(int).tolist()
            unique_classes.update(unique_values)

        expected_classes = self.expected_classes
        """
        assert (sorted(unique_classes) == expected_classes), \
            f"Expected classes {expected_classes}, but got {sorted(unique_classes)}"
        """

        logging.info("Dataset Information:")
        logging.info(f"  Number of samples: {self.num_entries}")
        logging.info(f"  Has priors: {self.haspriors()}")        
        logging.info(f"  Number of unique classes: {len(unique_classes)}")
        logging.info(f"  Unique class values: {sorted(unique_classes)}")
        logging.info(f"  Segmentation labels: {expected_classes}")
        logging.info(f"  Input shape: {list(input_shape[1:])}")
        logging.info(f"  Number of channels: {input_shape[0]}")
    
        return input_shape, unique_classes, label_lookup
