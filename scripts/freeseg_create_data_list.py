#!/usr/bin/env python

import os
import csv
import yaml
import argparse
from sklearn.model_selection import train_test_split


def create_dataset_file(
    data_folder, output_file=None, format=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
):
    image_paths = []
    label_paths = []

    images_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "labels")

    for filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, filename)
        label_path = os.path.join(
            labels_folder, filename.replace("img", "lbl")
        )  # Adjusted to your naming convention
        image_paths.append(image_path)
        label_paths.append(label_path)

    # Split dataset into training, validation, and testing
    train_img, test_img, train_lbl, test_lbl = train_test_split(
        image_paths, label_paths, test_size=(val_ratio + test_ratio), random_state=42
    )
    val_img, test_img, val_lbl, test_lbl = train_test_split(
        test_img, test_lbl, test_size=test_ratio / (test_ratio + val_ratio), random_state=42
    )

    # Determine the output format and filename
    if not output_file:
        output_file = os.path.join(
            data_folder, "dataset_list.yaml" if format == "yaml" else "dataset_list.csv"
        )

    if not format:
        _, ext = os.path.splitext(output_file)
        format = ext[1:].lower()  # Ensure format is in lowercase for comparison

    with open(output_file, "w", newline="") as f:
        if format == "csv":
            writer = csv.writer(f)
            writer.writerow(["image_filepath", "label_filepath", "split"])
            for img, lbl in zip(train_img, train_lbl):
                writer.writerow([img, lbl, "train"])
            for img, lbl in zip(val_img, val_lbl):
                writer.writerow([img, lbl, "validation"])
            for img, lbl in zip(test_img, test_lbl):
                writer.writerow([img, lbl, "test"])
        elif format == "yaml":
            data_split = {
                "train": [
                    {"image_filepath": img, "label_filepath": lbl}
                    for img, lbl in zip(train_img, train_lbl)
                ],
                "validation": [
                    {"image_filepath": img, "label_filepath": lbl}
                    for img, lbl in zip(val_img, val_lbl)
                ],
                "test": [
                    {"image_filepath": img, "label_filepath": lbl}
                    for img, lbl in zip(test_img, test_lbl)
                ],
            }
            yaml.dump(data_split, f)
        else:
            raise ValueError("Invalid format. Choose 'csv' or 'yaml'.")

    print(f"Dataset file created at: {output_file} with train, validation, and test splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and split image segmentation dataset file."
    )
    parser.add_argument("-d", "--data_folder", required=True, help="Path to the data folder.")
    parser.add_argument(
        "-o",
        "--output_file",
        help="Path to the output file (optional, defaults to 'dataset_list').",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Ratio of the training dataset."
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Ratio of the validation dataset."
    )
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of the test dataset.")
    args = parser.parse_args()

    create_dataset_file(
        args.data_folder,
        args.output_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
