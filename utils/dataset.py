import torch
import yaml
from torch.utils.data import Dataset
from utils.preprocessing import apply_augmentations
from utils.data_utils import load_volume
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the log level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print log messages to the console
    ],
)


class SegmentationDataset(Dataset):
    def __init__(self, dataset_entries, transform=None, crop_size=None):
        self.dataset_list = dataset_entries
        self.transform = transform
        self.crop_size = crop_size

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
        image, image_tensor = load_volume(image_path)
        label, label_tensor = load_volume(label_path)

        # Apply data augmentation if transform is specified
        if self.transform:
            image_tensor, label_tensor = apply_augmentations(
                image_tensor,
                label_tensor,
                image,
                label,
                voxsize=image.geom.voxsize,
                crop_size=self.crop_size,
                output_dir=None,
                augmentations_to_apply=self.transform,
            )
        
        return image_tensor, label_tensor

    def get_all_labels(self):
        all_labels = []
        for i in range(len(self)):
            _, labels = self[i]
            all_labels.append(labels)
        return torch.cat(all_labels, dim=0)


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
        transform=train_augmentations,
        crop_size=config["preprocessing"].get("crop_size"),
    )
    validation_dataset = SegmentationDataset(
        dataset_dict["validation"],
        transform=validation_augmentations,
        crop_size=config["preprocessing"].get("crop_size"),
    )
    test_dataset = SegmentationDataset(
        dataset_dict["test"],
        transform=test_augmentations,
        crop_size=config["preprocessing"].get("crop_size"),
    )

    return train_dataset, validation_dataset, test_dataset
