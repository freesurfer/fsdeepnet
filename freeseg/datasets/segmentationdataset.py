import os
import logging
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from freeseg.augmentation import apply_augmentations
from freeseg.utils import load_framedimage, save_framedimage, remap_labels, onehot

class SegmentationDataset(Dataset):
    def __init__(self, config, dataset_dict=None, image=None, label=None, transform=None, device=None, check_augment=False):
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
        self.ndims = config["model"]["ndims"]
        self.config = config
        self.augment_para = config["preprocessing"]
        self.transform = transform
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = len(sorted(config["dataset"]["expected_classes"]))
        self.label_mapping = config["dataset"]["label_mapping"]
        self.check_augment = check_augment

        assert (self.ndims == 3 or self.ndims == 2), "Model supports 3D or 2D"

        # Extract image and label file paths
        if (dataset_dict is not None):
            self.image_files = [item["image_filepath"] for item in dataset_dict]
            self.label_files = [item["label_filepath"] for item in dataset_dict]
        elif (image is not None and label is not None):
            self.image_files = image
            self.label_files = label

        assert (len(self.image_files) == len(self.label_files)), "image and label need to be the same length"


    def __len__(self):
        return self.num_entries

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]

        # Load image and label using the load_framedimage function
        image, image_tensor, _ = load_framedimage(image_path, orientation="RAS", device=self.device, ndims=self.ndims)
        label, label_tensor, _ = load_framedimage(label_path, orientation="RAS", device=self.device, ndims=self.ndims)

        # where/whether to save preprocessed data
        save_volumes = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = self.augment_para.get("augmentation_dir", None)
        if ((output_dir is not None) and (not os.path.exists(output_dir))):
            os.makedirs(output_dir)            

        # Apply data augmentation if transform is specified
        if self.transform:
            trycount = 1 if (self.check_augment) else None
            while (True):
                # image.geom.voxsize returned from surfa.load_volume() is (3, 1)
                # extract voxsizes to match {image_tensor.ndim-1}D data
                # make it writeable or voxynth.augment.image_augment() will complain non-writable numpy array
                voxsize = np.copy(image.geom.voxsize[:image_tensor.ndim-1])
                augmented_image_tensor, augmented_label_tensor = apply_augmentations(
                    image_tensor,
                    label_tensor,
                    image,
                    label,
                    self.config["dataset"].get("expected_classes"),
                    self.augment_para,
                    voxsize=voxsize,
                    output_dir=output_dir,
                    save_volumes=save_volumes + f"_try{trycount}" if (trycount is not None) else save_volumes,
                    augmentations_to_apply=self.transform,
                    left_right_corresponding=self.config["dataset"].get(
                        "left_right_corresponding", None
                    ),
                    device=self.device
                )
                
                # check if augmented label contains all the labels                             
                # compare the voxel counts of all labels
                havealllabels = True                               
                if (self.check_augment):
                    if (torch.count_nonzero(augmented_label_tensor) < torch.count_nonzero(label_tensor)):
                        if (output_dir is not None):
                            out_image = os.path.join(output_dir, save_volumes + f"_rejected{trycount}_image.mgz")
                            save_framedimage(augmented_image_tensor, out_image, original_framedimage=image)
                            out_label = os.path.join(output_dir, save_volumes + f"_rejected{trycount}_label.mgz")
                            save_framedimage(augmented_label_tensor, out_label, original_framedimage=label)
                        havealllabels = False                        
            
                if (havealllabels):
                    # freeseg.utils.remap_labels() and freeseg.utils.onehot() expect batched tensor [N, 1, H, W(, D)]
                    # add batch axis before calling remap_labels() and onehot()
                    augmented_label_tensor = augmented_label_tensor.int().unsqueeze(0)
                    onehot_augmented_label_tensor = remap_labels(augmented_label_tensor, self.label_mapping)
                    onehot_augmented_label_tensor = onehot(onehot_augmented_label_tensor, num_classes=self.num_classes, device=self.device)
                    # remove the added batch axis, DataLoader will batch the tensor based on batch_size
                    onehot_augmented_label_tensor = onehot_augmented_label_tensor.squeeze(0)
                    
                    if (output_dir is not None):
                        out_image = os.path.join(output_dir, save_volumes + f"_augmented_image.mgz")
                        save_framedimage(augmented_image_tensor, out_image, original_framedimage=image)
                        out_label = os.path.join(output_dir, save_volumes + f"_augmented_label.mgz")
                        save_framedimage(augmented_label_tensor, out_label, original_framedimage=label)
                        out_label_onehot = os.path.join(output_dir, save_volumes + f"_augmented_label_onehot.mgz")
                        save_framedimage(onehot_augmented_label_tensor, out_label_onehot, onehotencoded=True)
                    break

                trycount = trycount + 1
                logging.info(f"Reject {save_volumes} augmentation, retry #{trycount} ...")               

        return index, augmented_image_tensor, onehot_augmented_label_tensor

    def preload(self):
        """preprocesses all label maps, retrieve input tensor shape and unique classes."""

        logging.info("Perform dataset checking ...")

        expected_num_channels =self.config["dataset"]["expected_num_channels"]

        label_lookup = None
        unique_classes = set()
        for f_label, f_image in zip(self.label_files, self.image_files):
            label, label_tensor, _ = load_framedimage(f_label, device=self.device, ndims=self.ndims)
            image, image_tensor, _ = load_framedimage(f_image, device=self.device, ndims=self.ndims)

            # label_tensor and image_tensor are non-batched [C, H, W (,D)]
            assert (self.ndims == label_tensor.ndim - 1), f"Expected {self.ndims}D label, but got {label_tensor.ndim - 1}D"            
            assert (self.ndims == image_tensor.ndim - 1), f"Expected {self.ndims}D image, but got {image_tensor.ndim - 1}D"

            input_shape = image_tensor.shape
            assert (input_shape[0] == expected_num_channels), \
                f"Expected {expected_num_channels} channels, but got {input_shape[0]}"
            
            if (label_lookup is None):
                label_lookup = image.labels if (image.labels is not None) else label.labels

            unique_values = np.unique(label.data).astype(int).tolist()
            unique_classes.update(unique_values)

        expected_classes = self.config["dataset"]["expected_classes"]
        assert (sorted(unique_classes) == expected_classes), \
            f"Expected classes {expected_classes}, but got {sorted(unique_classes)}"

        logging.info("Dataset Information:")
        logging.info(f"  Number of samples: {self.num_entries}")
        logging.info(f"  Number of unique classes: {len(unique_classes)}")
        logging.info(f"  Unique class values: {sorted(unique_classes)}")
        logging.info(f"  Input shape: {input_shape[1:]}")
        logging.info(f"  Number of channels: {input_shape[0]}")
    
        return input_shape, unique_classes, label_lookup


def load_datasets(
    config,
    train_augmentations=None,
    validation_augmentations=None,
    test_augmentations=None,
    device=None,
    check_augment=False    
):
    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config["dataset"]["dataset_list_file"], "r") as file:
        dataset_dict = yaml.safe_load(file)

    dataset = dataset_dict.get("train")
    train_dataset = None
    if (dataset is not None):
        train_dataset = SegmentationDataset(
            config,
            dataset_dict=dataset,            
            transform=train_augmentations,
            device=device,
            check_augment=check_augment
        )

    dataset = dataset_dict.get("validation")
    validation_dataset = None
    if (dataset is not None):
        validation_dataset = SegmentationDataset(
            config,
            dataset_dict=dataset,            
            transform=validation_augmentations,
            device=device
        )

    dataset = dataset_dict.get("test")
    test_dataset = None
    if (dataset is not None):
        test_dataset = SegmentationDataset(
            config,
            dataset_dict=dataset,            
            transform=test_augmentations,
            device=device
        )

    return train_dataset, validation_dataset, test_dataset
