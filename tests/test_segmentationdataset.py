#!/usr/bin/env python

import os
import sys
import torch
import logging
import argparse
import numpy as np
import shutil

from freeseg.utils import load_config, set_deterministic_training
from freeseg.datasets import load_datasets, SegmentationDataset

"""
Usage: test_preprocessing.py 
       --config <config.yaml>
       [--augment]
       [--deterministic]
       [--outdir <augmentation_output_dir>]
       [--check_augment]
       [--image <im1 im2 ...> --label <lb1 lb2 ...> [--priors <...>]]
       [--dataset_list_file <dataset_list_file>]
       [--crop_size <W H D>]
       [--batch_size <n>]
       [--cpu]
       [--verbose]

    * - The input image/label is taken from one of the following:
        1. '--image <im1 im2 ...> --label <lb1 lb2 ...>'
        2. '--dataset_list_file <dataset_list_file>' or 
           config.yaml entry ["dataset"]["dataset_list_file"]
      - If augment = False (default), perform dataset checking only, no data augmentation.
"""

# Configure logging settings
logging.basicConfig(
    level=logging.DEBUG,  # Set the log level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to terminal
    ],
)


def main():
    logging.info("CWD: " + os.getcwd())
    logging.info(' '.join(sys.argv))
    
    args = argument_parse()

    if (args.cpu):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessing_device = device

    # Load config file
    config = load_config(args.config)

    # overwrite config with command line options
    if (args.dataset_list_file is not None):
        config["dataset"]["dataset_list_file"] = args.dataset_list_file    
    if (args.crop_size is not None):
        config["preprocessing"]["crop_size"] = args.crop_size
    if (args.deterministic is not None):
        config["training"]["deterministic"] = args.deterministic        
    if (args.batch_size is not None):
        config["training"]["batch_size"] = args.batch_size
    if (args.outdir is not None):
        config["preprocessing"]["augmentation_dir"] = args.outdir
    if (args.verbose):
        config["preprocessing"]["verbose"] = args.verbose
        
    # Access updated configuration values
    crop_size = config["preprocessing"]["crop_size"]
    deterministic = config["training"].get("deterministic", False)
    if (deterministic):
        # ??? todo: for multi-process dataloader, use worker_init_fn() and generator to preserve reproducibility
        #           see https://pytorch.org/docs/stable/notes/randomness.html
        set_deterministic_training()

    output_folder = config["preprocessing"].get("augmentation_dir", None)
    if (args.augment):
        assert (output_folder is not None), "Need to specify augmentation output directory"
        if (not os.path.exists(output_folder)):
            os.makedirs(output_folder)    

        # save config and dataset_list_file
        shutil.copyfile(args.config, os.path.join(output_folder, "config.yaml"))
        if (args.image is None or args.label is None):
            assert (config["dataset"].get("dataset_list_file", None) is not None), \
                "No input images are available. Use '--dataset_list_file <dataset_list_file>' or " \
                "'--image <im1 im2 ...> --label <lb1 lb2 ...>' to specify dataset."
        if (config["dataset"].get("dataset_list_file", None) is not None):
            shutil.copyfile(config["dataset"]["dataset_list_file"], os.path.join(output_folder, "dataset_list.yaml"))
        
    # create training dataset with the desired augmentations specified
    labels_segmentation = sorted(config["dataset"]["expected_classes"])
    label_mapping = {label:i for i, label in enumerate(labels_segmentation)}
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    config["dataset"]["label_mapping"] = label_mapping
    augmentation_class = config["preprocessing"].get("augmentation_class", "freeseg.augmentation.augmentbase.AugmentBase")
    if (args.image is not None and args.label is not None):
        logging.info("Loading dataset: SegmentationDataset(...)")
        train_dataset = SegmentationDataset(config, augmentation_class,
                                            image=args.image, label=args.label, priors=args.priors,
                                            transform=config["preprocessing"].get("train_augmentations"), device=preprocessing_device, check_augment=args.check_augment)
    else:
        logging.info("Loading dataset: load_dataset(...)")
        train_dataset, _, _ = load_datasets(config, augmentation_class,
            config["preprocessing"].get("train_augmentations"), config["evaluation"].get("evaluation_augmentations"), device=preprocessing_device, check_augment=args.check_augment)    

    sample_input_shape, unique_classes, label_lookup = train_dataset.preload()
    input_shape = sample_input_shape[1:]

    logging.info("Training Device: {}".format(device))
    logging.info("Preprocessing Device: {}".format(preprocessing_device))
    logging.info(f"Preprocessing augmentation_class: {augmentation_class}")
    logging.info(f"Preprocessing train_augmentations: {config['preprocessing'].get('train_augmentations')}")
    logging.info(f"batch_size: {config['training']['batch_size']}")
    logging.info(f"crop_size: {crop_size}")
    logging.info(f"deterministic: {deterministic}")
    logging.info(f"sampling_hyperparameters: {config['preprocessing'].get('sampling_hyperparameters', True)}")
    
    if (args.augment):
        logging.info("Perform data augmentation ...")        
        logging.info(f"Augmentation Output: {output_folder}")
        logging.info(f"training config: saved as {output_folder}/config.yaml")
        logging.info(f"dataset list: saved as {output_folder}/dataset_list.yaml")

        # output segmentation_labels.npy in training directory
        f_segmentation_labels = os.path.join(output_folder, "segmentation_labels.npy")
        np.save(f_segmentation_labels, np.array(sorted(unique_classes)).astype(int))
        
        for idx in range(len(train_dataset)):
            index, image_tensor, onehot_label_tensor, priors_tensor = train_dataset[idx]

    
def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--augment", action='store_true', help="Perform augmentation on input image/label.")
    parser.add_argument("--deterministic", action='store_true', help="deterministic training")
    parser.add_argument("--outdir", type=str, help="Path to augmentation output (needed for augmenting)")
    parser.add_argument("--check_augment", action='store_true', help="Reject augmentations not having all the labels")
    parser.add_argument("--image", nargs="+", type=str, help="Input image volume(s)")
    parser.add_argument("--label", nargs="+", type=str, help="Input label map(s)")
    parser.add_argument("--priors", nargs="+", type=str, help="Input priors")
    parser.add_argument("--dataset_list_file", type=str, help="Path to the dataset list file")
    parser.add_argument("--crop_size", nargs="+", type=int, help="Crop size for training and validation")
    parser.add_argument("--batch_size", type=int, help="Batch size for DataLoader")
    parser.add_argument("--cpu", action='store_true', help="Run on CPU.")
    parser.add_argument("--verbose", action='store_true', help="Print debug info to stdout")

    # parse commandline
    args = parser.parse_args()

    return args



# execute script
if __name__ == '__main__':
    main()
