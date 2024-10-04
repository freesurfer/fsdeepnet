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
    def __init__(self, dataset_entries, config, transform=None, device=None):
        self.dataset_list = dataset_entries
        self.config = config
        self.augment_para = config["preprocessing"]
        self.transform = transform
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape = None
        self.unique_classes = None

        # Extract image and label file paths
        self.image_files = [item["image_filepath"] for item in self.dataset_list]
        self.label_files = [item["label_filepath"] for item in self.dataset_list]

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        data_item = self.dataset_list[index]  # data_item is dict
        image_path = data_item.get("image_filepath")
        label_path = data_item.get("label_filepath")

        # Load image and label using the load_volume function
        image, image_tensor, _ = load_volume(image_path, orientation="RAS", device=self.device)
        label, label_tensor, _ = load_volume(label_path, orientation="RAS", device=self.device)

        # where/whether to save preprocessed data
        save_volumes = os.path.basename(image_path)
        output_dir = self.augment_para.get("augmentation_dir", None)

        # Apply data augmentation if transform is specified
        if self.transform:
            image_tensor, label_tensor = apply_augmentations(
                image_tensor,
                label_tensor,
                image,
                label,
                self.config["dataset"].get("expected_classes"),
                self.augment_para,
                voxsize=image.geom.voxsize,
                output_dir=output_dir,
                save_volumes=save_volumes,
                augmentations_to_apply=self.transform,
                left_right_corresponding=self.config["dataset"].get(
                    "left_right_corresponding", None
                ),
                device=self.device
            )

        return image_tensor, label_tensor

    def preload(self):
        """preprocesses all label maps, retrieve input tensor shape and unique classes."""
        self.unique_classes = set()

        for f_label, f_image in zip(self.label_files, self.image_files):
            label, label_tensor, _ = load_volume(f_label, device=self.device)
            image, image_tensor, _ = load_volume(f_image, device=self.device)

            if self.input_shape is None:
                self.input_shape = image_tensor.shape

            unique_values = np.unique(label.data).tolist()
            self.unique_classes.update(unique_values)

        return self.input_shape, self.unique_classes


    # test routines
    def test_preprocessing(self, outdir, augmentations=None):
        for idx in range(len(self.image_files)):
            f_image = self.image_files[idx]
            image, image_tensor, _ = load_volume(f_image, orientation="RAS", device=self.device)
            prefix = os.path.basename(f_image)
            reoriented = os.path.join(outdir, prefix + "_reoriented_image.mgz")
            save_volume(image_tensor, image, reoriented)

            f_label = self.label_files[idx]
            label, label_tensor, _ = load_volume(f_label, orientation="RAS", device=self.device)
            prefix = os.path.basename(f_label)
            reoriented = os.path.join(outdir, prefix + "_reoriented_label.mgz")
            save_volume(label_tensor, label, reoriented)

            if augmentations is not None:
                print(f"Augmentations to apply: {augmentations}")
                prefix = os.path.basename(f_image)
                image_tensor, label_tensor = apply_augmentations(
                    image_tensor,
                    label_tensor,
                    image,
                    label,
                    self.config["dataset"].get("expected_classes"),
                    self.augment_para,
                    voxsize=image.geom.voxsize,
                    output_dir=outdir,
                    save_volumes=prefix,
                    augmentations_to_apply=augmentations,
                    left_right_corresponding=self.config["dataset"].get(
                        "left_right_corresponding", None
                    ),
                    device=self.device
                )


def load_datasets(
    config,
    train_augmentations=None,
    validation_augmentations=None,
    test_augmentations=None,
    device=None
):
    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config["dataset"]["dataset_list_file"], "r") as file:
        dataset_dict = yaml.safe_load(file)

    dataset = dataset_dict.get("train")
    train_dataset = None
    if (dataset is not None):
        train_dataset = SegmentationDataset(
            dataset,
            config,
            transform=train_augmentations,
            device=device
        )

    dataset = dataset_dict.get("validation")
    validation_dataset = None
    if (dataset is not None):
        validation_dataset = SegmentationDataset(
            dataset,
            config,
            transform=validation_augmentations,
            device=device
        )

    dataset = dataset_dict.get("test")
    test_dataset = None
    if (dataset is not None):
        test_dataset = SegmentationDataset(
            dataset,
            config,
            transform=test_augmentations,
            device=device
        )

    return train_dataset, validation_dataset, test_dataset


def dataGenerator(dataloader, device=None):
    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while (True):
        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device).float(), labels.to(device)

            yield idx, images, labels
