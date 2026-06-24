#!/usr/bin/env python

import os
import sys
import csv
import yaml
import argparse
from sklearn.model_selection import train_test_split


"""
Usage: freeseg_split_dataset.py
       [--d <data_folder> | --i <dataset.yaml>]
       [--o <output_file>]
       [--ignore_prior]
       [--train_ratio <train_ratio>]
       [--val_ratio <val_ratio>]
       [--test_ratio <test_ratio>]

    * defaults: output_file='dataset_list.yaml', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    * <data_folder> is assumed to have the following directory structure:
          data_folder/
          |---------- images/ (optional)
          |---------- labels/
          |---------- priors/ (optional)
      images, labels, priors are placed under their corresponding directories with same filename for each subject.
    * <dataset.yaml> is expected to have all data assigned to 'dataset'
    * --d <data_folder> and --i <dataset.yaml> are mutually exclusive
"""

def main():
    print(' '.join(sys.argv))

    args = argument_parse()

    assert (not (args.d and args.i)), "Options '--i' and '--d' are mutually exclusive"
    assert (args.d or args.i), "Need input, use '--i <dataset.yaml>', or '--d <data_folder>'"

    if (args.i):
        vols, segs, priors = read_dataset_yaml(args.i)
    else:
        vols, segs, priors = read_datafolder(args.d)

    assert (len(segs) > 0), "Empty label dataset"
    
    if (len(vols) == 0):
        vols = [None] * len(segs)
    if (len(priors) == 0):
        priors = [None] * len(segs)

    assert (len(vols) == len(segs)),   "'vols' and 'segs' need to be the same length"
    assert (len(priors) == len(segs)), "'priors' and 'segs' need to be the same length"

    out_split = args.o
    if (out_split is None):
        out_split = os.path.join(os.getcwd(), "dataset_list.yaml")
        
    create_split_dataset(vols, segs, priors, out_split, args.ignore_prior,
                         train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)


def create_split_dataset(vols, segs, priors, out_split, ignore_prior=False,
                         train_ratio=None, val_ratio=None, test_ratio=None):
    # set default train/val/test split ratio if none is specified
    if (train_ratio is None and val_ratio is None and test_ratio is None):
        train_ratio = 0.7
        val_ratio   = 0.15
        test_ratio  = 0.15

    split_ratios  = [train_ratio, val_ratio, test_ratio]
    # convert float list to string
    split_ratios = [str(x) if x is not None else str(0.0) for x in split_ratios]
    split_subsets = []

    # Split dataset into training, validation, and testing
    if (train_ratio is not None and val_ratio is not None and test_ratio is not None):
        split_subsets = ['train', 'validation', 'test']
        train_vols, test_vols, train_segs, test_segs, train_priors, test_priors = \
            train_test_split(vols, segs, priors, test_size=(val_ratio+test_ratio), random_state=42)
        val_vols, test_vols, val_segs, test_segs, val_priors, test_priors = \
            train_test_split(test_vols, test_segs, test_priors, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
    elif (train_ratio is not None and val_ratio is not None):
        split_subsets = ['train', 'validation']
        train_vols, val_vols, train_segs, val_segs, train_priors, val_priors = \
            train_test_split(vols, segs, priors, test_size=val_ratio, random_state=42)        
    elif (val_ratio is not None and test_ratio is not None):
        split_subsets = ['validation', 'test']
        val_vols, test_vols, val_segs, test_segs, val_priors, test_priors = \
            train_test_split(vols, segs, priors, test_size=test_ratio, random_state=42)
    else:
        errmsg = f"Unsupported train/validation/test split ratio {':'.join(split_ratios)}"
        raise ValueError(errmsg)
    
    # output split dataset
    trainset, valset, testset = [], [], []
    if (train_ratio is not None):
        for vol, seg, prior in zip(train_vols, train_segs, train_priors):
            update_dset(trainset, vol, seg, prior)
    if (val_ratio is not None):
        for vol, seg, prior in zip(val_vols, val_segs, val_priors):
            update_dset(valset, vol, seg, prior)
    if (test_ratio is not None):
        for vol, seg, prior in zip(test_vols, test_segs, test_priors):
            update_dset(testset, vol, seg, prior)

    # save split dataset
    data_split = {}
    if (len(trainset) > 0):
        data_split["train"] = trainset
    if (len(valset) > 0):
        data_split["validation"] = valset
    if (len(testset) > 0):
        data_split["test"] = testset

    # output split dataset as yaml
    with open(out_split, "w", newline="") as f:            
        yaml.dump(data_split, f)

    print(f"The dataset is split into " + '/'.join(split_subsets) + " subsets with " + ':'.join(split_ratios) + " ratios.")
    print(f"Split dataset is saved as {out_split}")


# read vol/seg/prior in the data folder
# the data folder is expected to have the following directory structure:
#          data_folder/
#          |---------- images/ (optional)
#          |---------- labels/
#          |---------- priors/ (optional)
# images, labels, priors are placed under their corresponding directories with same filename for each subject.
def read_datafolder(datafolder):
    datafolder = os.path.abspath(datafolder)
    
    vols, segs, priors = [], [], []
    segs_folder = os.path.join(datafolder, "labels")    
    for filename in os.listdir(segs_folder):
        segs.append(os.path.join(datafolder, "labels", filename))

        vol = os.path.join(datafolder, "images", filename)
        if (os.path.isfile(vol)):
            vols.append(vol)

        prior = os.path.join(datafolder, "priors", filename)
        if (os.path.isfile(prior)):
            priors.append(prior)

    return vols, segs, priors


# all vol/seg/prior pairs are expected to assigned to 'dataset' in the input dataset.yaml
# only 'label_filepath' is required
def read_dataset_yaml(in_dataset_yaml):
    with open(in_dataset_yaml, "r", encoding='utf-8') as file:
        dataset_dict = yaml.safe_load(file)
    dataset = dataset_dict.get("dataset")

    vols, segs, priors = [], [], []
    for item in dataset:
        segs.append(item.get("label_filepath"))
        
        vol = item.get("image_filepath")
        if (vol is not None):
            vols.append(vol)
            
        prior = item.get("prior_filepath")
        if (prior is not None):
            priors.append(prior)

    return vols, segs, priors


def update_dset(dict, vol, seg, prior):
    entry = {}
    if (vol is not None):
        entry.update({"image_filepath": vol})
    entry.update({"label_filepath": seg})
    if (prior is not None):
        entry.update({"prior_filepath": prior})

    dict.append(entry)


def argument_parse():
    # command line parser
    parser = argparse.ArgumentParser(description="Create and split image segmentation dataset file.")

    # command line arguments
    parser.add_argument("--d", help="Path to the data folder.")
    parser.add_argument("--i", help="Path to the input dataset.yaml.")
    parser.add_argument("--o", help="Path to the output file (optional, defaults to 'dataset_list.yaml').")
    parser.add_argument("--ignore_prior", action='store_true', help="Don't include priors in output dataset yaml")
    parser.add_argument("--train_ratio", type=float, help="Ratio of the training dataset.")
    parser.add_argument("--val_ratio", type=float, help="Ratio of the validation dataset.")
    parser.add_argument("--test_ratio", type=float, help="Ratio of the test dataset.")

    # parse command line arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
