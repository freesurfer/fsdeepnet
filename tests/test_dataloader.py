#!/usr/bin/env python

import os
import sys
import torch
import logging
import argparse
import datetime
import numpy as np
import shutil

from torch.utils.data import DataLoader

from freeseg.utils import load_config, DataGenerator, set_deterministic_training
from freeseg.datasets import load_datasets


"""
Usage: test_dataloader.py 
       --config <config.yaml>
       [--deterministic]
       [--dataset_list_file <dataset_list_file>]
       [--train_output_folder <train_output_folder>]
       [--crop_size <W H D>]
       [--cpu]
       [--num_workers <num_workers>]
       [--prefetch_factor <prefetch_factor>]
       [--pin_memory]
       [--persistent_workers]

    * The input image/label is taken from '--dataset_list_file <dataset_list_file>' or config.yaml entry ["dataset"]["dataset_list_file"]
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
    if (args.train_output_folder is not None):
        config["training"]["train_output_folder"] = args.train_output_folder
    if (args.deterministic is not None):
        config["training"]["deterministic"] = args.deterministic         
    if (args.num_workers is not None):
        config["preprocessing"]["num_workers"] = args.num_workers
    if (args.prefetch_factor is not None):
        config["preprocessing"]["prefetch_factor"] = args.prefetch_factor        
    if (args.pin_memory is not None):
        config["preprocessing"]["pin_memory"] = args.pin_memory
    if (args.persistent_workers is not None):
        config["preprocessing"]["persistent_workers"] = args.persistent_workers        

    train_output_folder = config["training"].get("train_output_folder", None)
    assert (train_output_folder is not None), "Use '--train_output_folder <>' or 'train_output_folder' in config.yaml to specify training output directory"
    assert (config["dataset"].get("dataset_list_file", None) is not None), \
        "no dataset_list_file is available. Use '--dataset_list_file <dataset_list_file>' to specify dataset."

    output_folder = os.path.abspath(train_output_folder)
    if (not os.path.exists(output_folder)):
        os.makedirs(output_folder)
    
    # save config and dataset_list_file
    shutil.copyfile(args.config, os.path.join(output_folder, "config.yaml"))
    shutil.copyfile(config["dataset"]["dataset_list_file"], os.path.join(output_folder, "dataset_list.yaml"))

    # Access updated configuration values
    deterministic = config["training"]["deterministic"]
    if (deterministic):
        set_deterministic_training()    
    crop_size = config["preprocessing"]["crop_size"]
    num_workers = config["preprocessing"].get("num_workers", 0)
    pin_memory = config["preprocessing"].get("pin_memory", False)
    persistent_workers = config["preprocessing"].get("persistent_workers", False)
    if (num_workers == 0):
        prefetch_factor = config["preprocessing"].get("prefetch_factor", None)
        persistent_workers = False
    else:
        prefetch_factor = config["preprocessing"].get("prefetch_factor", 2)

    # force data preprocessing (augmentation) to run on CPU is pin_memory = True or num_worker > 0
    if (pin_memory or num_workers > 0):
        preprocessing_device = torch.device("cpu")
        
    # create training dataset with the desired augmentations specified
    logging.info("Loading dataset: load_dataset(...)")
    labels_segmentation = sorted(config["dataset"]["expected_classes"])
    label_mapping = {label:i for i, label in enumerate(labels_segmentation)}
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    config["dataset"]["label_mapping"] = label_mapping
    augmentation_class = config["preprocessing"].get("augmentation_class", "freeseg.augmentation.augmentbase.AugmentBase")
    train_dataset, _, _ = load_datasets(config, augmentation_class,
                                        config["preprocessing"].get("train_augmentations"), config["evaluation"].get("evaluation_augmentations"), device=preprocessing_device)

    # Create training DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True,
                              pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

    # ??? todo: we probably can get rid of sample_input_shape too
    sample_input_shape, unique_classes, label_lookup = train_dataset.preload()
    input_shape = sample_input_shape[1:]

    # output segmentation_labels.npy in training directory
    f_segmentation_labels = os.path.join(output_folder, "segmentation_labels.npy")
    np.save(f_segmentation_labels, np.array(sorted(unique_classes)).astype(int))
    
    logging.info("Training Device: {}".format(device))
    logging.info("Preprocessing Device: {}".format(preprocessing_device))
    logging.info(f"Preprocessing augmentation_class: {augmentation_class}")
    logging.info(f"Preprocessing train_augmentations: {config['preprocessing'].get('train_augmentations')}")
    logging.info(f"Preprocessing pin_memory: {pin_memory}")
    logging.info(f"Preprocessing num_workers: {num_workers}")
    logging.info(f"Preprocessing prefetch_factor: {prefetch_factor}")
    logging.info(f"Preprocessing persistent_workers: {persistent_workers}")

    logging.info(f"train_output_folder: {output_folder}")
    logging.info(f"crop_size: {crop_size}")
    logging.info(f"deterministic: {deterministic}")
    logging.info(f"training config: saved as {output_folder}/config.yaml")
    logging.info(f"dataset list: saved as {output_folder}/dataset_list.yaml")

    train_dataset_dict = {
        "batch_size": config["training"]["batch_size"],
        "segmentation_labels": sorted(unique_classes),
        "crop_size": crop_size,
        "num_samples": len(train_dataset),
        "input_shape": input_shape,
        "num_channels": sample_input_shape[0],        
    }
    
    labels_segmentation = np.array(sorted(unique_classes)).astype(int)
    num_labels = len(labels_segmentation)
    label_mapping = {label.item(): i for i, label in enumerate(labels_segmentation)}
        
    start_epoch = 0
    epochs = config["training"]["dice_epochs"]
    input_generator = DataGenerator(train_loader, preprocessing_device)
    steps_per_epoch = config["training"]["steps_per_epoch"]    
    for epoch in range(start_epoch, epochs):
        logging.info(f"Epoch {epoch+1:3d}/{epochs:<3d}")
        for step in range(steps_per_epoch):
            (batch_idx, images, onehot_labels, priors, dataset_indices) = next(input_generator)
            logging.info(f"  {step+1:4d}/{steps_per_epoch:<4d} preprocessed batch #{batch_idx:<2d}, (training set index {dataset_indices.tolist()})")
            #torch.cuda._sleep(500)

                       
    
def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--deterministic", action='store_true', help="deterministic training")
    parser.add_argument("--dataset_list_file", type=str, help="Path to the dataset list file")
    parser.add_argument("--train_output_folder", type=str, default=None, help="Base folder for saving training outputs")    
    parser.add_argument("--cpu", action='store_true', help="Run on CPU.")
    parser.add_argument("--num_workers", type=int, help="Number of Dataloader workers")
    parser.add_argument("--prefetch_factor", type=int, help="Number of batches loaded in advance by each worker")
    parser.add_argument("--pin_memory", action='store_true', help="Store data in pinned memory")
    parser.add_argument("--persistent_workers", action='store_true', help=" Keep the workers Dataset instances alive")
    parser.add_argument("--crop_size", nargs="+", type=int, help="Crop size for training and validation")

    # parse commandline
    args = parser.parse_args()

    return args


# execute script
if __name__ == '__main__':
    main()
