#!/usr/bin/env python

import os
import sys
import torch
import logging
import argparse
import numpy as np
import shutil

from freeseg.training import Training
from freeseg.config import Config

"""
Usage: test_preprocessing.py 
       --config <config.yaml>
       [--augment]
       [--deterministic]
       [--augmentation_dir <augmentation_output_dir>]
       [--dataset_list_file <dataset_list_file> --cohort <train|validation|test>]
       [--crop_size <W H D>]
       [--batch_size <n>]
       [--cpu]
       [--verbose]

    * - The input image/label is taken from '--dataset_list_file <dataset_list_file>' or 
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
    args = argument_parse()
    
    config = Config.process(args, logger=logging, require_train_outfolder=False, test_augment=args.augment)
    config, _, _, _, _, train_dataset = Training.setup(config, preload_dataset=args.preload, create_loader=False, create_model=False)
    Config.print(config, logging)

    if (args.augment):
        logging.info("Perform data augmentation ...")
        for idx in range(len(train_dataset)):
            index, image_tensor, onehot_label_tensor, priors_tensor = train_dataset[idx]

    
def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--preload", action='store_true', help="Preload training dataset")
    parser.add_argument("--augment", action='store_true', help="Perform augmentation on input image/label.")
    parser.add_argument("--deterministic", action='store_true', help="deterministic training")
    parser.add_argument("--augmentation_dir", type=str, help="Path to augmentation output (needed for augmenting)")
    parser.add_argument("--dataset_list_file", type=str, help="Path to the dataset list file")
    parser.add_argument("--train_cohort", nargs="+", type=str, default=['train'], help="Specify dataset cohort. Can be combinations of train, validation, or test")
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
