import os
import pytest
from utils.data_utils import load_config, load_volume, save_volume
from utils.preprocessing import apply_augmentations
from utils.dataset import load_datasets

# Load configuration
config = load_config("configs/config.yaml")

# Load datasets
train_dataset, _, _ = load_datasets(config)

# Set number of volumes to augment and desired augmentations
num_volumes_to_augment = 2
augmentations_to_apply = ["cropping",]

# Define output directory (user-defined)
output_dir = "output/tests"
os.makedirs(output_dir, exist_ok=True)


@pytest.mark.parametrize("index", range(num_volumes_to_augment))
def test_augmentations_and_saving(index):
    # Get image and label paths from the training dataset
    image_path = train_dataset.image_files[index]
    label_path = train_dataset.label_files[index]

    # Load image and label
    image, image_tensor = load_volume(image_path)
    label, label_tensor = load_volume(label_path)

    # Extract basename and extension from image path
    basename, ext = os.path.splitext(os.path.basename(image_path))

    # Apply specified augmentations
    augmented_image, augmented_label = apply_augmentations(
        image_tensor,
        label_tensor,
        image,
        label,
        image.geom.voxsize,
        output_dir=None,  # No output directory needed
        save_volumes=False,  # We'll handle saving manually
        augmentations_to_apply=augmentations_to_apply,
    )

    # Construct output filenames with the original basename and augmentation information
    for augmentation in augmentations_to_apply:
        aug_image_filename = f"{basename}_{augmentation}_image{ext}"
        aug_label_filename = f"{basename}_{augmentation}_label{ext}"

        # Save augmented volumes in the user-defined output directory
        save_volume(
            augmented_image, image, os.path.join(output_dir, aug_image_filename)
        )
        save_volume(
            augmented_label, label, os.path.join(output_dir, aug_label_filename)
        )

    # Print message indicating where the augmented volumes are saved
    print(f"Augmented volumes for {basename} saved in: {output_dir}")
