import os
import logging
import numpy as np
import torch

from freeseg import augmentation
from freeseg.utils import utility as utils

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dset_profile, augment_obj, dataset_dict, device=None, keep_trainset_in_memory=False, preload=False, augdir=None):
        """
        SegmentationDataset Constructor

        dataset_dict : dict
          Input dataset dict containing input image volumes and label maps
        """

        self.dset_profile = dset_profile
        self.num_entries = len(dataset_dict)
        self.num_channels = dset_profile["expected_num_channels"]
        self.ndims = dset_profile["ndims"]
        self.num_classes = dset_profile["num_labels"]
        self.label_mapping = dset_profile["label_mapping"]

        assert (self.ndims == 3 or self.ndims == 2), "Model supports 3D or 2D"
                
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if (keep_trainset_in_memory and augdir is not None):
            errmsg = f"'--keep_trainset_in_memory' doesn't work with saving augmentation volumes"
            logging.error(errmsg) 
            raise ValueError(errmsg)
        if (keep_trainset_in_memory and not preload):
            errmsg = f"'--keep_trainset_in_memory' only work with 'preload=True'"
            logging.error(errmsg) 
            raise ValueError(errmsg)

        self.keep_trainset_in_memory = keep_trainset_in_memory
        self.images, self.labels = [], []
        self.image_tensors , self.label_tensors, self.prior_tensors = [], [], []
        
        # Extract image, label, priors file paths
        self.image_files, self.label_files, self.priors_files = [], [], []
        if (dataset_dict is not None):
            for item in dataset_dict:
                if (item.get("image_filepath")):
                    self.image_files.append(item["image_filepath"])
                self.label_files.append(item["label_filepath"])
                if (item.get("prior_filepath")):
                    self.priors_files.append(item["prior_filepath"])

        if (self.hasimage()):
            assert (len(self.image_files) == len(self.label_files)), "image and label need to be the same length"
        if (self.haspriors()):
            assert (len(self.label_files) == len(self.priors_files)), "label and priors need to be the same length"

        self.data_augment = augment_obj
        if (self.data_augment is not None):
            augmentation.check_augmentations(self.data_augment)

        # load first label
        label0, label_tensor0, _ = utils.load_framedimage(self.label_files[0], orientation="RAS", device=self.device, ndims=self.ndims)
        self.target_res = label0.geom.voxsize[:self.ndims]
        self.label_lookup = label0.labels
        self.dset_profile.update({"num_samples": self.num_entries,
                                  "input_shape": list(label_tensor0.shape[1:]),
                                  "target_res": self.target_res,
                                  "num_channels": self.num_channels,
                                  "image": self.hasimage(),
                                  "priors": self.haspriors(),
                                  "label_lookup": self.label_lookup})          
        logging.info("")
        logging.info("Dataset Information:")
        logging.info(f"  Number of samples: {self.num_entries}")
        logging.info(f"  Has image: {self.hasimage()}")
        logging.info(f"  Has priors: {self.haspriors()}")        
        logging.info(f"  Input shape: {list(label_tensor0.shape[1:])}")
        logging.info(f"  Input resolution: {self.target_res}")
        logging.info(f"  Number of channels: {self.num_channels}")
        if (preload):
            generation_labels = self.preload()
            self.dset_profile.update({"reported_generation_labels": generation_labels})


    def hasimage(self):
        return True if (len(self.image_files) > 0) else False


    def haspriors(self):
        return True if (len(self.priors_files) > 0) else False

    
    def __len__(self):
        return self.num_entries


    def __getitem__(self, index):
        image_path = self.image_files[index] if (self.hasimage()) else None        
        label_path = self.label_files[index]

        if (not self.keep_trainset_in_memory):
            image, image_tensor, priors_tensor = None, None, None            
            # load label
            label, label_tensor, _ = utils.load_framedimage(label_path, orientation="RAS", device=self.device, ndims=self.ndims)
            assert (label_tensor.shape[0] == self.num_channels), \
                f"Expected {self.num_channels} channels, but got {label_tensor.shape[0]}"
            assert (np.all(label.geom.voxsize[:self.ndims] == self.target_res)), \
                f"Expected resolution {self.target_res}mm, but got {label.geom.voxsize}mm"
            # load image if they are provided
            if (self.hasimage()):
                image, image_tensor, _ = utils.load_framedimage(image_path, orientation="RAS", device=self.device, ndims=self.ndims)
                assert (label_tensor.shape == image_tensor.shape), \
                    f"image and label need to be in the same shape. label {f_label} has shape {label_tensor.shape}, image {f_image} has shape {image_tensor.shape}"                
                assert (np.all(label.geom.voxsize == image.geom.voxsize)), \
                    f"image and label need to have the same resolution. label {f_label} is {label.geom.voxsize}mm, image {f_image} is {image.geom.voxsize}mm"
            # load priors if they are provided
            if (self.haspriors()):
                priors_path = self.priors_files[index]        
                sfprior, priors_tensor, _  = utils.load_framedimage(priors_path, orientation="RAS", device=self.device, ndims=self.ndims)
        else:
            # retrieve preloaded data
            # saving augmentated volumes will not work when keep_trainset_in_memory=True
            # because the input volume names are not available
            sfprior = None
            image = self.images[index]
            label = self.labels[index]            
            image_tensor  = self.image_tensors[index]
            label_tensor  = self.label_tensors[index]
            priors_tensor = self.prior_tensors[index]

        label_tensor  = label_tensor.int()
        if (image_tensor is not None):
            image_tensor  = image_tensor.float()
        if (priors_tensor is not None):
            priors_tensor = priors_tensor.float()

        # Apply data augmentation
        if (self.data_augment is not None):
            augmented_image_tensor, augmented_label_tensor, augmented_priors_tensor = \
                augmentation.apply_augmentations(
                    self.data_augment,
                    image_tensor,
                    label_tensor,
                    image,
                    label,
                    priors_tensor=priors_tensor,
                    orig_fpath=image_path if (self.hasimage()) else label_path,
                    index=index
                )

            # freeseg.utils.remap_labels() and freeseg.utils.onehot() expect batched tensor [N, 1, H, W(, D)]
            # add batch axis before calling remap_labels() and onehot()
            augmented_label_tensor = augmented_label_tensor.int().unsqueeze(0)
            onehot_augmented_label_tensor = utils.remap_labels(augmented_label_tensor, self.label_mapping)
            onehot_augmented_label_tensor = utils.onehot(onehot_augmented_label_tensor, num_classes=self.num_classes, device=self.device)
            # remove the added batch axis, DataLoader will batch the tensor based on batch_size
            onehot_augmented_label_tensor = onehot_augmented_label_tensor.squeeze(0)

            if (augmented_priors_tensor is None):
                # torch.utils.data.DataLoader can't return NoneType, make an empty tensor with 0 elements
                augmented_priors_tensor = torch.empty(0, *onehot_augmented_label_tensor.shape[1:], device=self.device)

            return index, augmented_image_tensor, onehot_augmented_label_tensor, augmented_priors_tensor
        else:
            # freeseg.utils.remap_labels() and freeseg.utils.onehot() expect batched tensor [N, 1, H, W(, D)]
            # add batch axis before calling remap_labels() and onehot()
            onehot_label_tensor = utils.remap_labels(label_tensor.int().unsqueeze(0), self.label_mapping)
            onehot_label_tensor = utils.onehot(onehot_label_tensor, num_classes=self.num_classes, device=self.device)
            # remove the added batch axis, DataLoader will batch the tensor based on batch_size
            onehot_label_tensor = onehot_label_tensor.squeeze(0)
            if (priors_tensor is None):
                # torch.utils.data.DataLoader can't return NoneType, make an empty tensor with 0 elements
                priors_tensor = torch.empty(0, *onehot_label_tensor.shape[1:], device=self.device)
            
            return index, image_tensor, onehot_label_tensor, priors_tensor


    def preload(self):
        """preprocesses all label maps, retrieve input tensor shape and unique classes."""

        logging.info("Perform dataset checking ...")

        expected_num_channels = self.num_channels

        generation_labels = set()
        for n in range(self.num_entries):
            ### load labels
            f_label = self.label_files[n]
            label, label_tensor, _ = utils.load_framedimage(f_label, orientation="RAS", device=self.device, ndims=self.ndims)

            # label_tensor is non-batched [C, H, W (,D)]
            assert (self.ndims == label_tensor.ndim - 1), f"Expected {self.ndims}D label, but got {label_tensor.ndim - 1}D"
            input_shape = label_tensor.shape            

            ### load images
            image, image_tensor = None, None
            if (self.hasimage()):
                f_image = self.image_files[n]
                # image_tensor is non-batched [C, H, W (,D)]
                image, image_tensor, _ = utils.load_framedimage(f_image, orientation="RAS", device=self.device, ndims=self.ndims)
                assert (label_tensor.shape == image_tensor.shape), \
                    f"image and label need to be in the same shape. label {f_label} has shape {label_tensor.shape}, image {f_image} has shape {image_tensor.shape}"
                assert (np.all(label.geom.voxsize == image.geom.voxsize)), \
                    f"image and label need to have the same resolution. label {f_label} is {label.geom.voxsize}mm, image {f_image} is {image.geom.voxsize}mm"
                input_shape = image_tensor.shape # use image shape if it is available

            ### load priors
            prior, prior_tensor = None, None
            if (self.haspriors()):
                f_prior = self.priors_files[n]
                prior, prior_tensor, _ = utils.load_framedimage(f_prior, orientation="RAS", device=self.device, ndims=self.ndims)
                
                # prior_tensor is non-batched [self.num_classes, H, W (,D)]
                assert (list(prior_tensor.shape) == [self.num_classes, *label_tensor.shape[1:]]), \
                    f"Expected prior shape [self.num_classes, *label_tensor.shape[1:]], but got {list(prior_tensor.shape)}"              

            assert (input_shape[0] == expected_num_channels), \
                f"Expected {expected_num_channels} channels, but got {input_shape[0]}"
            if (self.target_res is None):
                self.target_res = label.geom.voxsize[:self.ndims]
            else:
                assert (np.all(label.geom.voxsize[:self.ndims] == self.target_res)), \
                    f"Expected resolution {self.target_res}mm, but got {label.geom.voxsize}mm"

            if (self.keep_trainset_in_memory):
                self.images.append(image)
                self.labels.append(label)
                self.image_tensors.append(image_tensor)
                self.label_tensors.append(label_tensor)
                self.prior_tensors.append(prior_tensor)

            if (self.label_lookup is None):
                self.label_lookup = image.labels if (image is not None and image.labels is not None) else label.labels

            ### collect all the labels in the dataset
            unique_labels = np.unique(label.data).astype(int).tolist()
            generation_labels.update(unique_labels)

        logging.info(f"  Reported number of labels: {len(generation_labels)}")
        logging.info(f"  Reported generation labels: {sorted(generation_labels)}")
    
        return generation_labels

    @property
    def profile(self):
        return self.dset_profile
