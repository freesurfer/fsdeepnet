#!/usr/bin/env python

import os
import torch
import logging
import argparse
import numpy as np
import shutil

from freeseg.utils import load_config
from freeseg.datasets import load_datasets, SegmentationDataset

"""
Usage: test_preprocessing.py 
       --config <config.yaml>
       [--augment]
       [--outdir <augmentation_output_dir>]
       [--image <im1 im2 ...> --label <lb1 lb2 ...>]
       [--dataset_list_file <dataset_list_file>]
       [--crop_size <W H D>]
       [--num_iters <n>]
       [--cpu]
       [--debug]

    * - The input image/label is taken from
        1. '--image <im1 im2 ...> --label <lb1 lb2 ...>'
        2. '--dataset_list_file <dataset_list_file>' or 
           config.yaml entry ["dataset"]["dataset_list_file"]
      - If augment = False (default), perform dataset checking only, no data augmentation.
"""

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,  # Set the log level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to terminal
    ],
)


def main():
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
    if (args.outdir is not None):
        config["preprocessing"]["augmentation_dir"] = args.outdir
    if (args.debug):
        config["preprocessing"]["debug"] = args.debug
        
    # Access updated configuration values
    crop_size = config["preprocessing"]["crop_size"]

    output_folder = config["preprocessing"].get("augmentation_dir", None)
    if (args.augment):
        assert (output_folder is not None), "Need to specify augmentation output directory"
        if (not os.path.exists(output_folder)):
            os.makedirs(output_folder)    

        # save config and dataset_list_file
        shutil.copyfile(args.config, os.path.join(output_folder, "config.yaml"))
        shutil.copyfile(config["dataset"]["dataset_list_file"], os.path.join(output_folder, "dataset_list.yaml"))
        
    # create training dataset with the desired augmentations specified
    if (args.image is not None and args.label is not None):
        logging.info("Loading dataset: SegmentationDataset(...)")
        train_dataset = SegmentationDataset(config, image=args.image, label=args.label, transform=config["preprocessing"].get("train_augmentations"), device=preprocessing_device)
    else:
        logging.info("Loading dataset: load_dataset(...)")
        train_dataset, _, _ = load_datasets(
            config, config["preprocessing"].get("train_augmentations"), config["evaluation"].get("evaluation_augmentations"), device=preprocessing_device
        )    

    # ??? todo: we probably can get rid of sample_input_shape too
    sample_input_shape, unique_classes, label_lookup = train_dataset.preload()
    input_shape = sample_input_shape[1:]

    if (args.augment):
        # output segmentation_labels.npy in training directory
        f_segmentation_labels = os.path.join(output_folder, "segmentation_labels.npy")
        np.save(f_segmentation_labels, np.array(sorted(unique_classes)).astype(int))
    
    logging.info("Training Device: {}".format(device))
    logging.info("Preprocessing Device: {}".format(preprocessing_device))
    logging.info(f"crop_size: {crop_size}")
    
    if (args.augment):
        logging.info("Perform data augmentation ...")        
        logging.info(f"Augmentation Output: {output_folder}")
        logging.info(f"training config: saved as {output_folder}/config.yaml")
        logging.info(f"dataset list: saved as {output_folder}/dataset_list.yaml")

        for niter in range(args.num_iters):
            for idx in range(len(train_dataset)):
                index, image_tensor, label_tensor = train_dataset[idx]

    
def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--augment", action='store_true', help="Perform augmentation on input image/label.")        
    parser.add_argument("--outdir", type=str, help="Path to augmentation output (needed for augmenting)")
    parser.add_argument("--image", nargs="+", type=str, help="Input image volume(s)")
    parser.add_argument("--label", nargs="+", type=str, help="Input label map(s)")
    parser.add_argument("--dataset_list_file", type=str, help="Path to the dataset list file")
    parser.add_argument("--crop_size", nargs="+", type=int, help="Crop size for training and validation")
    parser.add_argument("--num_iters", type=int, default=1, help="How many augmentation iterations")
    parser.add_argument("--cpu", action='store_true', help="Run on CPU.")
    parser.add_argument("--debug", action='store_true', help="Output debug information/volume")

    # parse commandline
    args = parser.parse_args()

    return args



# execute script
if __name__ == '__main__':
    main()
