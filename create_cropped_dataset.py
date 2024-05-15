import os
import yaml
from utils.data_utils import load_volume, save_volume
from utils.preprocessing import apply_cropping


def create_cropped_dataset(dataset_list_file, crop_size=(80, 80, 80), output_folder="cropped_data"):
    """
    Creates a combined cropped dataset with separate image/label folders from the dataset_list.yaml file.

    Args:
        dataset_list_file (str): Path to the dataset_list.yaml file.
        crop_size (tuple, optional): The desired crop size (depth, height, width). Defaults to (80, 80, 80).
        output_folder (str, optional): The folder to save the cropped dataset. Defaults to "cropped_data".
    """

    # Load dataset list from YAML file
    with open(dataset_list_file, "r") as file:
        dataset_dict = yaml.safe_load(file)

    # Get image and label paths from dataset dict
    image_paths = []
    label_paths = []
    for dataset_type in ["train", "validation", "test"]:
        dataset_list = dataset_dict[dataset_type]
        image_paths.extend([item["image_filepath"] for item in dataset_list])
        label_paths.extend([item["label_filepath"] for item in dataset_list])

    # Create output folders for images and labels
    images_output_folder = os.path.join(output_folder, "images")
    labels_output_folder = os.path.join(output_folder, "labels")
    os.makedirs(images_output_folder, exist_ok=True)
    os.makedirs(labels_output_folder, exist_ok=True)

    # Process image and label paths
    for image_path, label_path in zip(image_paths, label_paths):
        # Load image and label
        image, image_tensor = load_volume(image_path)
        label, label_tensor = load_volume(label_path)

        # Apply cropping
        cropped_image, cropped_label = apply_cropping(image_tensor, label_tensor, crop_size)

        # Extract basename from paths
        image_basename = os.path.basename(image_path)
        label_basename = os.path.basename(label_path)

        # Construct output paths for cropped volumes
        cropped_image_path = os.path.join(images_output_folder, image_basename)
        cropped_label_path = os.path.join(labels_output_folder, label_basename)

        # Save cropped volumes
        save_volume(cropped_image, image, cropped_image_path)
        save_volume(cropped_label, label, cropped_label_path)

    print(f"Cropped datasets saved in: {output_folder}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create combined cropped dataset with separate image/label folders from dataset_list.yaml."
    )
    parser.add_argument(
        "-d",
        "--dataset_list_file",
        required=True,
        help="Path to the dataset_list.yaml file.",
    )
    parser.add_argument(
        "-s",
        "--crop_size",
        type=int,
        nargs=3,
        default=[80, 80, 80],
        help="The desired crop size (depth, height, width).",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        default="cropped_data",
        help="The folder to save the cropped dataset.",
    )
    args = parser.parse_args()

    create_cropped_dataset(args.dataset_list_file, args.crop_size, args.output_folder)
