import os
import argparse
import yaml

from utils.dataset import SegmentationDataset
from utils.data_utils import load_volume


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test data augmentations on a sample image and label."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/augmentation_tests",
        help="Directory to save the augmented images.",
    )
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load a sample image and label
    dataset_list_file = config["dataset"]["dataset_list_file"]
    with open(dataset_list_file, "r") as file:
        dataset_dict = yaml.safe_load(file)

    sample_data = dataset_dict["train"][0]  # Take the first sample from the training set
    image_path = sample_data["image_filepath"]
    label_path = sample_data["label_filepath"]

    # Load using the load_volume function from your data_utils.py
    image, image_tensor = load_volume(image_path)
    label, label_tensor = load_volume(label_path)

    # Create a SegmentationDataset instance (you might need to adjust the arguments)
    dataset = SegmentationDataset(
        dataset_entries=[sample_data],  # Pass a list containing the sample data
        config=config,
        transform=config["preprocessing"].get("train_augmentations"),
    )

    # Test and save augmented images
    # dataset.test_preprocessing(outdir=args.output_dir, augmentations=dataset.transform)
    dataset.test_individual_augmentation(outdir=args.output_dir, augmentations=dataset.transform)

    print(f"Augmented images saved to: {args.output_dir}")