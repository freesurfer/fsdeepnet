#!/usr/bin/env python

import sys
import logging
import argparse

from freeseg.utils import utility as utils
from freeseg.training import Training
from freeseg.config import Config

"""
Usage: test_dataloader.py 
       --config <config.yaml>
       [--deterministic]
       [--dataset_list_file <dataset_list_file> --train_cohort <train|validation|test>]
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
    args = argument_parse()
    
    config = Config.process(args, logger=logging, require_train_outfolder=False)
    config, train_loader, _, _, _, _, _ = Training.setup(config, preload_dataset=False, create_val_loader=False, create_model=False)
    Config.print(config, logging)

    start_epoch = 0
    epochs = config["training"]["dice_epochs"]
    steps_per_epoch = config["training"]["steps_per_epoch"]

    # retrieve data_generator information
    cfg_data_generator = config["training"].pop("data_generator", {})
    if (cfg_data_generator):
        data_generator = utils.get_class(cfg_data_generator.pop("fn", "freeseg.utils.utility.DataGenerator"))
    else:
        data_generator = utils.get_class("freeseg.utils.utility.DataGenerator")
    cfg_data_generator.update({"device" : config["preprocessing_device"]})
    input_generator = data_generator(train_loader, **cfg_data_generator)
    
    for epoch in range(start_epoch, epochs):
        logging.info(f"Epoch {epoch+1:3d}/{epochs:<3d}")
        for step in range(steps_per_epoch):
            batched_sample = next(input_generator)

            batch_idx = batched_sample.pop(0)         # remove first item batch_idx
            dataset_indices = batched_sample.pop(-1)  # remove last item dataset_indices
            batch_indices = ", ".join(str(item).zfill(4) for item in dataset_indices.tolist())
            hasextra = True if (len(batched_sample) == 3) else False
            if (hasextra):
                images, labels, extra = batched_sample
            else:
                images, labels = batched_sample

            msg = f"  {step+1:4d}/{steps_per_epoch:<4d} batch #{batch_idx:<2d} ({batch_indices}), images({images.shape}), labels({labels.shape}) "
            if (hasextra):
                shape = len(extra) if(isinstance(extra, list)) else extra.shape
                msg += f", extra({shape})"
            logging.info(msg)
            #torch.cuda._sleep(500)

    
def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--deterministic", action='store_true', help="deterministic training")
    parser.add_argument("--dataset_list_file", type=str, help="Path to the dataset list file")
    parser.add_argument("--train_cohort", nargs="+", type=str, default=['train'], help="Specify dataset cohort. Can be combinations of train, validation, or test")
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
