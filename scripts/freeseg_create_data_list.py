#!/usr/bin/env python

import os
import sys
import csv
import yaml
import argparse
from sklearn.model_selection import train_test_split


"""
Usage: freeseg_create_data_list.py
       -d <data_folder>
       [-o <output_file>]
       [--ignore_prior]
       [--train_ratio <train_ratio>]
       [--val_ratio <val_ratio>]
       [--test_ratio <test_ratio>]

    * defaults: output_file='dataset_list.yaml', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    * the following directory structure is assumed:
          data_folder/
          |---------- images/
          |---------- labels/
          |---------- priors/
    * images, labels, priors are placed under their corresponding directories with same filename for each subject.
"""

def main():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser(
        description="Create and split image segmentation dataset file."
    )
    parser.add_argument("-d", "--data_folder", required=True, help="Path to the data folder.")
    parser.add_argument(
        "-o",
        "--output_file",
        help="Path to the output file (optional, defaults to 'dataset_list.yaml').",
    )
    parser.add_argument("--ignore_prior", action='store_true', help="Don't include priors in output dataset yaml")
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
        args.ignore_prior,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

def create_dataset_file(data_folder, output_file=None, ignore_prior=False,
                        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    images, labels = [], []
    labels_folder = os.path.join(data_folder, "labels")    
    for filename in os.listdir(labels_folder):
        images.append(filename)
        labels.append(filename)

    # Split dataset into training, validation, and testing
    train_img, test_img, train_lbl, test_lbl = train_test_split(
        images, labels, test_size=(val_ratio + test_ratio), random_state=42
    )
    val_img, test_img, val_lbl, test_lbl = train_test_split(
        test_img, test_lbl, test_size=test_ratio / (test_ratio + val_ratio), random_state=42
    )
    train_pri, val_pri, test_pri = train_lbl.copy(), val_lbl.copy(), test_lbl.copy()
    
    # retrieve image, label, prior full path
    train_lbl = [os.path.join(labels_folder, lbl) for lbl in train_lbl]
    test_lbl = [os.path.join(labels_folder, lbl) for lbl in test_lbl]
    val_lbl = [os.path.join(labels_folder, lbl) for lbl in val_lbl]

    images_folder = os.path.join(data_folder, "images")
    train_img = [os.path.join(images_folder, img) for img in train_img]
    val_img = [os.path.join(images_folder, img) for img in val_img]
    test_img = [os.path.join(images_folder, img) for img in test_img]

    priors_folder = os.path.join(data_folder, "priors")
    if (not ignore_prior and os.path.isdir(priors_folder)):
        train_pri = [os.path.join(priors_folder, lbl) for lbl in train_pri]
        val_pri = [os.path.join(priors_folder, lbl) for lbl in val_pri]
        test_pri = [os.path.join(priors_folder, lbl) for lbl in test_pri]
    else:
        train_pri, val_pri, test_pri = [None]*len(train_lbl), [None]*len(val_lbl), [None]*len(test_lbl)

    # Determine the output format and filename
    if (not output_file):
        output_file = os.path.join(data_folder, "dataset_list.yaml")

    with open(output_file, "w", newline="") as f:
        trainset, valset, testset = [], [], []
        for img, lbl, pri in zip(train_img, train_lbl, train_pri):
            entry = {"image_filepath": img, "label_filepath": lbl}
            if (pri is not None):
                entry.update({"prior_filepath": pri})
            trainset.append(entry)
        for img, lbl, pri in zip(val_img, val_lbl, val_pri):
            entry = {"image_filepath": img, "label_filepath": lbl}
            if (pri is not None):
                entry.update({"prior_filepath": pri})
            valset.append(entry)
        for img, lbl, pri in zip(test_img, test_lbl, test_pri):
            entry = {"image_filepath": img, "label_filepath": lbl}
            if (pri is not None):
                entry.update({"prior_filepath": pri})
            testset.append(entry)            
        data_split = {"train": trainset, "validation": valset, "test": testset}
        yaml.dump(data_split, f)

    print(f"Dataset file created at: {output_file} with train, validation, and test splits.")


if __name__ == "__main__":
    main()
