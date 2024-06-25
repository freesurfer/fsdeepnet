import os
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from utils.preprocessing import (
    apply_augmentations,
    apply_flipping,
    apply_spatial_transform,
    apply_cropping,
    apply_blur_resample,
    apply_bias_field,
)
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
        self.augment_para = config["preprocessing"]
        self.transform = transform

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
        image, image_tensor = load_volume(image_path, orientation="RAS")
        label, label_tensor = load_volume(label_path, orientation="RAS")

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
            )

        return image_tensor, label_tensor

    def preload(self):
        """preprocesses all label maps, retrieve input tensor shape and unique classes."""
        self.unique_classes = set()
        all_labels = []

        for f_label, f_image in zip(self.label_files, self.image_files):
            label, label_tensor = load_volume(f_label)
            image, image_tensor = load_volume(f_image)

            if self.input_shape is None:
                self.input_shape = image_tensor.shape  # This should be (2, H, W, D)

            print(f"[debug - dataset] Preloaded image shape: {image_tensor.shape}")
            print(f"[debug - dataset] Preloaded label shape: {label_tensor.shape}")

            unique_values = np.unique(label.data).tolist()
            self.unique_classes.update(unique_values)
            all_labels.append(label_tensor)

        return self.input_shape, self.unique_classes, torch.cat(all_labels, dim=0)

    """
    # the functionality is merged into preload()
    def get_all_labels(self):
        all_labels = []
        for i in range(len(self)):
            _, labels = self[i]
            all_labels.append(labels)
        return torch.cat(all_labels, dim=0)
    """

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
                )

    def test_individual_augmentation(self, outdir, augmentations=None):
        for idx in range(len(self.image_files)):
            f_image = self.image_files[idx]
            prefix = os.path.basename(f_image)

            for augmentation_name in augmentations:
                # --- Reload Original Volumes Before Each Augmentation ---
                image, image_tensor = load_volume(f_image, orientation="RAS")
                label, label_tensor = load_volume(self.label_files[idx], orientation="RAS")

                # --- Apply Augmentations ---
                if augmentation_name == "flipping":
                    flip_prob = self.augment_para.get("flip_prob")
                    aff = image.geom.vox2world.matrix
                    image_tensor, label_tensor = apply_flipping(
                        image_tensor,
                        label_tensor,
                        aff,
                        self.config["dataset"].get("left_right_corresponding", None),
                        flip_prob,
                    )
                    save_volumes = prefix + "_flipped"  # Create filename with augmentation name

                elif augmentation_name == "spatial_transform":
                    image_tensor, label_tensor = apply_spatial_transform(
                        image_tensor,
                        label_tensor,
                        image.geom.voxsize,
                        affine_probability=self.augment_para.get("affine_probability", 1.0),
                        max_translation=self.augment_para.get("max_translation", 5.0),
                        max_rotation=self.augment_para.get("max_rotation", 5.0),
                        max_scaling=self.augment_para.get("max_scaling", 1.1),
                        warp_probability=self.augment_para.get("warp_probability", 1.0),
                        warp_integrations=self.augment_para.get("warp_integrations", 7),
                        warp_smoothing_range=self.augment_para.get(
                            "warp_smoothing_range", [10, 20]
                        ),
                        warp_magnitude_range=self.augment_para.get("warp_magnitude_range", [1, 2]),
                        shearing_bounds=self.augment_para.get("shearing_bounds", 0.015),
                    )
                    save_volumes = prefix + "_transformed"

                elif augmentation_name == "cropping":
                    crop_size = self.augment_para.get("crop_size", None)
                    if crop_size is not None:
                        image_tensor, label_tensor = apply_cropping(
                            image_tensor, label_tensor, crop_size
                        )
                        save_volumes = prefix + "_cropped"
                    else:
                        raise ValueError(
                            "Crop size must be provided when using the 'cropping' augmentation."
                        )

                elif augmentation_name == "blur_resample":
                    image_tensor = apply_blur_resample(
                        image_tensor,
                        image.geom.voxsize,
                        smoothing_probability=self.augment_para.get("smoothing_probability", 0.5),
                        smoothing_max_sigma=self.augment_para.get("smoothing_max_sigma", 2.0),
                        added_noise_probability=self.augment_para.get(
                            "added_noise_probability", 0.5
                        ),
                        added_noise_max_sigma=self.augment_para.get("added_noise_max_sigma", 0.05),
                        gamma_scaling_probability=self.augment_para.get(
                            "gamma_scaling_probability", 0.5
                        ),
                        gamma_scaling_max=self.augment_para.get("gamma_scaling_max", 0.8),
                        resized_probability=self.augment_para.get("resized_probability", 0),
                        resized_one_axis_probability=self.augment_para.get(
                            "resized_one_axis_probability", 0
                        ),
                        resized_max_voxsize=self.augment_para.get("resized_max_voxsize", 2),
                    )
                    save_volumes = prefix + "_blur_resampled"

                elif augmentation_name == "bias_field":
                    image_tensor = apply_bias_field(
                        image_tensor,
                        image.geom.voxsize,
                        bias_field_probability=self.augment_para.get("bias_field_probability", 0.5),
                        bias_field_max_magnitude=self.augment_para.get(
                            "bias_field_max_magnitude", 0.1
                        ),
                        bias_field_smoothing_range=self.augment_para.get(
                            "bias_field_smoothing_range", [1, 2]
                        ),
                    )
                    save_volumes = prefix + "_bias_field"
                else:
                    raise ValueError(f"Unknown augmentation: {augmentation_name}")

                # --- Saving Augmented Volumes ---
                if save_volumes is not None and outdir is not None:
                    save_volume(
                        image_tensor,
                        image,
                        os.path.join(outdir, save_volumes + "_image.mgz"),
                    )
                    save_volume(
                        label_tensor,
                        label,
                        os.path.join(outdir, save_volumes + "_label.mgz"),
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
