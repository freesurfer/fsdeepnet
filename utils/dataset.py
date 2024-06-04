import os
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from utils.preprocessing import apply_augmentations
from utils.data_utils import load_volume, save_volume
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the log level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print log messages to the console
    ],
)


class SegmentationDataset(Dataset):
    def __init__(self, dataset_entries, config, transform=None):
        self.dataset_list = dataset_entries
        self.config = config
        self.transform = transform
        self.crop_size = self.config["preprocessing"].get("crop_size")

        self.input_shape = None
        self.unique_classes = None

        # Extract image and label file paths
        self.image_files = [item["image_filepath"] for item in self.dataset_list]
        self.label_files = [item["label_filepath"] for item in self.dataset_list]

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        data_item = self.dataset_list[index]
        image_path = data_item["image_filepath"]
        label_path = data_item["label_filepath"]

        # Load image and label using the load_volume function
        image, image_tensor = load_volume(image_path, orientation='RAS')
        label, label_tensor = load_volume(label_path, orientation='RAS')

        # where/whether to save preprocessed data
        save_volumes = os.path.basename(image_path)
        output_dir = None
        if (self.config["preprocessing"].get("augmentation_dir") != 'None'):
            output_dir = self.config["preprocessing"].get("augmentation_dir")

        # Apply data augmentation if transform is specified
        if self.transform:
            image_tensor, label_tensor = apply_augmentations(
                image_tensor,
                label_tensor,
                image,
                label,
                self.config,
                voxsize=image.geom.voxsize,
                output_dir=output_dir,
                save_volumes=save_volumes,
                augmentations_to_apply=self.transform,
            )

        return image_tensor, label_tensor

    # this method preprocesses all label maps, retrieve input tensor shape and unique classes
    def preload(self):
        self.unique_classes = set()
        # loop through self.label_files, get unique classes
        for f_label in self.label_files:
            label, label_tensor = load_volume(f_label)
            if (self.input_shape is None):
                self.input_shape = label_tensor.shape
                
            unique_values = np.unique(label.data).tolist()
            self.unique_classes.update(unique_values)

        return self.input_shape, self.unique_classes

    def get_all_labels(self):
        all_labels = []
        for i in range(len(self)):
            _, labels = self[i]
            all_labels.append(labels)
        return torch.cat(all_labels, dim=0)

    # test routines
    def test_preprocessing(self, outdir, augmentations=None):
        for idx in range(len(self.image_files)):
            f_image = self.image_files[idx]
            image, image_tensor = load_volume(f_image, orientation='RAS')
            prefix = os.path.basename(f_image)
            reoriented = os.path.join(outdir, prefix + "_reoriented_image.mgz")
            save_volume(
                image_tensor,
                image,
                reoriented
            )

            f_label = self.label_files[idx]
            label, label_tensor = load_volume(f_label, orientation='RAS')
            prefix = os.path.basename(f_label)
            reoriented = os.path.join(outdir, prefix + "_reoriented_label.mgz")
            save_volume(
                label_tensor,
                label,
                reoriented
            )            

            if (augmentations is not None):
                prefix = os.path.basename(f_image)
                image_tensor, label_tensor = apply_augmentations(
                    image_tensor,
                    label_tensor,
                    image,
                    label,
                    self.config,
                    voxsize=image.geom.voxsize,
                    output_dir=outdir,
                    save_volumes=prefix,
                    augmentations_to_apply=augmentations,
                )

            
def load_datasets(
    config,
    train_augmentations=None,
    validation_augmentations=None,
    test_augmentations=None,
):
    with open(config["dataset"]["dataset_list_file"], "r") as file:
        dataset_dict = yaml.safe_load(file)

    train_dataset = SegmentationDataset(
        dataset_dict["train"],
        config,
        transform=train_augmentations,
    )
    validation_dataset = SegmentationDataset(
        dataset_dict["validation"],
        config,
        transform=validation_augmentations,
    )
    test_dataset = SegmentationDataset(
        dataset_dict["test"],
        config,
        transform=test_augmentations,
    )

    return train_dataset, validation_dataset, test_dataset
