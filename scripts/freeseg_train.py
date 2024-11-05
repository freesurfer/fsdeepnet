#!/usr/bin/env python

import os
import torch
import logging
import argparse
import datetime
import numpy as np
import shutil

from torch.utils.data import DataLoader

from freeseg.models import UNet
from freeseg.training import Training
from freeseg.utils import load_config
from freeseg.datasets import load_datasets
from freeseg.metrics import WeightedL2Loss, DiceLoss, DiceScore

"""
Usage: train.py 
       --config <config.yaml>
       [--dataset_list_file <dataset_list_file>]
       [--ctab <ctab>]
       [--train_root_folder <train_root_folder>]
       [--run_name <--run_name>]
       [--checkpoint <checkpoint>]
       [--crop_size <W H D>]
       [--write_tensorboard_summary]
       [--perform_evaluation]
       [--best_model_metric <loss|dice>]
       [--cpu]
       [--num_workers <num_workers>]
       [--prefetch_factor <prefetch_factor>]
       [--pin_memory]
       [--persistent_workers]
       [--debug]
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

    checkpoint = args.checkpoint
    ctab = args.ctab
    
    # Load config file
    config = load_config(args.config)

    # overwrite config with command line options
    # train_root_folder and run_name are set to default
    train_root_folder = args.train_root_folder
    if (train_root_folder is None):
        train_root_folder = config.get("training", {}).get("train_root_folder", "new_runs/tensorboard_logs")
    config["training"]["train_root_folder"] = train_root_folder

    run_name  = args.run_name
    if (run_name is None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.get("training", {}).get("run_name", f"run_{timestamp}")
    config["training"]["run_name"] = run_name

    if (args.debug):
        config["preprocessing"]["debug"] = args.debug
    if (args.dataset_list_file is not None):
        config["dataset"]["dataset_list_file"] = args.dataset_list_file    
    if (args.crop_size is not None):
        config["preprocessing"]["crop_size"] = args.crop_size
    if (args.write_tensorboard_summary):
        config["training"]["write_tensorboard_summary"] = args.write_tensorboard_summary
    if (args.perform_evaluation):
        config["training"]["perform_evaluation"] = args.perform_evaluation
    if (args.best_model_metric is not None):
       config["training"]["best_model_metric"] = args.best_model_metric
    if (args.num_workers is not None):
        config["preprocessing"]["num_workers"] = args.num_workers
    if (args.prefetch_factor is not None):
        config["preprocessing"]["prefetch_factor"] = args.prefetch_factor        
    if (args.pin_memory is not None):
        config["preprocessing"]["pin_memory"] = args.pin_memory
    if (args.persistent_workers is not None):
        config["preprocessing"]["persistent_workers"] = args.persistent_workers        

    crop_size = config["preprocessing"]["crop_size"]
    nb_levels = config["model"]["nb_levels"]
    assert (np.all(np.array(crop_size) % (2**(nb_levels-1)) == 0)), f"crop_size {crop_size} needs to be divisible by 2^{nb_levels-1}"
    ndims = config["model"]["ndims"]
    assert (ndims == len(crop_size)), f"crop_size {crop_size} is not for {ndims}D"

    """
    # yaml has nested structure, the update doesn't update value in nested structure
    # Update configuration with command-line arguments
    config_updates = {k: v for k, v in vars(args).items() if v is not None}
    config.update(config_updates)
    """

    output_folder = os.path.join(train_root_folder, run_name)    # Output folder for current run
    if (not os.path.exists(output_folder)):
        os.makedirs(output_folder)

    # save config and dataset_list_file
    shutil.copyfile(args.config, os.path.join(output_folder, "config.yaml"))
    shutil.copyfile(config["dataset"]["dataset_list_file"], os.path.join(output_folder, "dataset_list.yaml"))

    # Access updated configuration values
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
        
    # create training/validation dataset with the desired augmentations specified
    train_dataset, validation_dataset, _ = load_datasets(
        config, config["preprocessing"].get("train_augmentations"), config["evaluation"].get("evaluation_augmentations"), device=preprocessing_device
    )

    # Create training DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True,
                              pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

    input_shape, unique_classes, label_lookup = train_dataset.preload()

    # output segmentation_labels.npy in training directory
    f_segmentation_labels = os.path.join(output_folder, "segmentation_labels.npy")
    np.save(f_segmentation_labels, np.array(sorted(unique_classes)).astype(int))
    
    # create validation DataLoader
    validation_loader = None
    perform_evaluation = config["training"].get("perform_evaluation", False)
    if (perform_evaluation):
        best_model_metric = config["training"]["best_model_metric"]
        validation_loader = DataLoader(validation_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    logging.info("Training Device: {}".format(device))
    logging.info("Preprocessing Device: {}".format(preprocessing_device))
    logging.info(f"Preprocessing pin_memory: {pin_memory}")
    logging.info(f"Preprocessing num_workers: {num_workers}")
    logging.info(f"Preprocessing prefetch_factor: {prefetch_factor}")
    logging.info(f"Preprocessing persistent_workers: {persistent_workers}")

    if (checkpoint is not None):
        logging.info(f"resume training from model: {checkpoint}")
    logging.info(f"train_root_folder: {train_root_folder}")
    logging.info(f"run_name: {run_name}")
    logging.info(f"crop_size: {crop_size}")
    logging.info(f"color table: {ctab}")
    if (perform_evaluation):
        logging.info(f"best_model_metric: {best_model_metric}")
    logging.info(f"training config: saved as {output_folder}/config.yaml")
    logging.info(f"dataset list: saved as {output_folder}/dataset_list.yaml")

    train_dataset_dict = {
        "batch_size": config["training"]["batch_size"],
        "segmentation_labels": sorted(unique_classes),
        "crop_size": crop_size,
        "num_samples": len(train_dataset),
        "input_shape": input_shape[1:],
        "num_channels": input_shape[0],        
    }

    train(train_loader, config, output_folder, len(unique_classes), ctab, label_lookup, checkpoint, validation_loader, device, preprocessing_device, train_dataset_dict, debug=args.debug)
                       
    
def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--dataset_list_file", type=str, help="Path to the dataset list file")
    parser.add_argument("--ctab", type=str, help="Path to the lookup table")
    parser.add_argument("--train_root_folder", type=str, default=None, help="Base folder for saving training outputs")    
    parser.add_argument("--run_name", type=str, default=None, help="Descriptive name for the run (used for naming TensorBoard log directories)")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint file to resume training from")
    parser.add_argument("--cpu", action='store_true', help="Run on CPU.")
    parser.add_argument("--num_workers", type=int, help="Number of Dataloader workers")
    parser.add_argument("--prefetch_factor", type=int, help="Number of batches loaded in advance by each worker")
    parser.add_argument("--pin_memory", action='store_true', help="Store data in pinned memory")
    parser.add_argument("--persistent_workers", action='store_true', help=" Keep the workers Dataset instances alive")
    parser.add_argument("--crop_size", nargs="+", type=int, help="Crop size for training and validation")
    #parser.add_argument("--expected_classes", nargs="+", type=int, help="Expected classes in the dataset")
    parser.add_argument("--write_tensorboard_summary", action='store_true', help="Write tensorboard summary")
    parser.add_argument("--perform_evaluation", action='store_true', help="Perform evaluation after each epoch")
    parser.add_argument("--best_model_metric", type=str, default=None, choices=["loss", "dice"], help="Metric for saving the best model (loss or dice)")
    parser.add_argument("--debug", action='store_true', help="Output volumes for debugging.")

    # parse commandline
    args = parser.parse_args()

    return args


def train(train_loader, config, train_output_folder, num_labels, ctab, label_lookup=None, checkpoint=None,
          validation_loader=None, device=None, preprocessing_device=None, train_dataset_dict=None, debug=False):
    # create the model to train
    model_arch_dict = config["model"]
    model_arch_dict["name"] = "UNet"
    model_arch_dict["num_channels"] = config["dataset"]["expected_num_channels"]
    model_arch_dict["nb_labels"] = len(config["dataset"]["expected_classes"])
    model_arch_dict["final_pred_activation"] = config["model"].get("final_pred_activation", "softmax")
    model = UNet(model_arch_dict).to(device)
   
    # print Model Architecture
    # from torchinfo import summary
    # input_shape = train_dataset_dict["input_shape"]
    # summary(model, input_size=input_shape)

    # create the Training object
    trainer = Training(train_output_folder,
                       config["dataset"]["expected_classes"],
                       train_loader,
                       model,
                       model_arch_dict=model_arch_dict,
                       train_dataset_dict=train_dataset_dict,
                       ctab=ctab,
                       label_lookup=label_lookup,
                       model_checkpoint=checkpoint,
                       validation_loader=validation_loader,
                       best_model_metric=config["training"]["best_model_metric"],
                       write_tensorboard_summary=config["training"].get("write_tensorboard_summary", False),
                       device=device,
                       preprocessing_device=preprocessing_device,
                       debug=debug)
                       
    # train wl2 epochs (??? todo: make this optional ???)
    wl2_loss_fn = WeightedL2Loss()
    trainer.train_model(lr=config["training"]["pre_train_learning_rate"],
                        epochs=config["training"]["wl2_epochs"],
                        steps_per_epoch=config["training"]["steps_per_epoch"],
                        metric_type='wl2',
                        loss_fn=wl2_loss_fn)
                       
    # train dice epochs
    dice_loss_fn = DiceLoss(
        num_classes=num_labels,
        input_type="prob",
        dice_type="soft"
    )                   
    trainer.train_model(lr=config["training"]["pre_train_learning_rate"],
                        epochs=config["training"]["dice_epochs"],
                        steps_per_epoch=config["training"]["steps_per_epoch"],
                        metric_type='dice',
                        loss_fn=dice_loss_fn)                   


# execute script
if __name__ == '__main__':
    main()
