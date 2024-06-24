import os
import torch
from torch.utils.data import Dataset
import yaml
from omegaconf import DictConfig
from utils.data_utils import load_volume, save_volume
from utils.preprocessing import apply_augmentations
import logging

log = logging.getLogger(__name__)

class SegmentationDataset(Dataset):
    def __init__(self, dataset_entries, cfg: DictConfig, transform=None):
        self.dataset_list = dataset_entries
        self.cfg = cfg
        self.transform = transform

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

        # Apply data augmentation if transform is specified
        if self.transform:
            image_tensor, label_tensor = apply_augmentations(
                image_tensor,
                label_tensor,
                image,
                label,
                self.cfg,
                augmentations_to_apply=self.transform
            )

    # test routines
    def test_preprocessing(self, outdir, augmentations=None):
        for idx in range(len(self.image_files)):
            f_image = self.image_files[idx]
            image, image_tensor = load_volume(f_image, orientation="RAS")
            prefix = os.path.basename(f_image)
            reoriented = os.path.join(outdir, prefix + "_reoriented_image.mgz")
            save_volume(image_tensor, image, reoriented)

            f_label = self.label_files[idx]
            label, label_tensor = load_volume(f_label, orientation="RAS")
            prefix = os.path.basename(f_label)
            reoriented = os.path.join(outdir, prefix + "_reoriented_label.mgz")
            save_volume(label_tensor, label, reoriented)

            if augmentations is not None:
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
                )

def load_datasets(cfg: DictConfig):
    with open(cfg.dataset.dataset_list_file, "r") as file:
        dataset_dict = yaml.safe_load(file)

    train_dataset = SegmentationDataset(
        dataset_dict["train"],
        cfg,
        transform=cfg.preprocessing.train_augmentations,
    )
    validation_dataset = SegmentationDataset(
        dataset_dict["validation"],
        cfg,
        transform=cfg.preprocessing.validation_augmentations,
    )
    test_dataset = SegmentationDataset(
        dataset_dict["test"],
        cfg,
        transform=cfg.preprocessing.test_augmentations,
    )

    return train_dataset, validation_dataset, test_dataset
