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

from freeseg.training import Training
from freeseg.config import Config
from freeseg.utils import set_deterministic_training, print_vm_peak, config_logger, get_class, remove_duplicates
from freeseg.datasets import load_datasets
from freeseg.metrics import WeightedL2Loss, DiceLoss

"""
Usage: freeseg_train.py 
       --config <config.yaml>
       [--train_output_folder <train_output_folder>]
       [--keep_trainset_in_memory]
       [--deterministic]
       [--model_name <model_classname>]
       [--dataset_list_file <dataset_list_file>]
       [--ctab <ctab>]
       [--checkpoint <checkpoint>]
       [--crop_size <W H D>]
       [--batch_size <n>]
       [--write_tensorboard_summary]
       [--perform_evaluation]
       [--best_model_metric <loss|dice>]
       [--cpu]
       [--num_workers <num_workers>]
       [--prefetch_factor <prefetch_factor>]
       [--pin_memory]
       [--persistent_workers]
       [--debug]
       [--vmp]
       [--verbose]
       [--logfile <logfile>]
"""

mainlogger = logging.getLogger(__name__)
mainlogger.addHandler(logging.StreamHandler())

def main():
    args = argument_parse()

    if (args.cpu):
        os.environ["CUDA_VISIBLE_DEVICES"]=""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_index = torch.cuda.current_device()
    else:
        device = torch.device("cpu")
        gpu_index = None
    preprocessing_device = device

    checkpoint = args.checkpoint
    ctab = args.ctab
    
    # Load config file
    config = Config.load(args.config)

    # overwrite config with command line options
    if (args.verbose):
        config["preprocessing"]["verbose"] = args.verbose
    if (args.model_name is not None):
        config["model"]["name"] = args.model_name
    if (args.weight_init is not None):
        config["model"]["weight_init"] = args.weight_init
    if (args.dataset_list_file is not None):
        config["dataset"]["dataset_list_file"] = args.dataset_list_file
    if (args.crop_size is not None):
        config["preprocessing"]["crop_size"] = args.crop_size
    if (args.train_output_folder is not None):
        config["training"]["train_output_folder"] = args.train_output_folder
    if (args.deterministic is not None):
        config["training"]["deterministic"] = args.deterministic
    if (args.batch_size is not None):
        config["training"]["batch_size"] = args.batch_size
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

    train_output_folder = config["training"].get("train_output_folder", None)
    assert (train_output_folder is not None), "Use '--train_output_folder <>' or 'train_output_folder' in config.yaml to specify training output directory"
    assert (config["dataset"].get("dataset_list_file", None) is not None), "Use '--dataset_list_file <dataset.yaml>' or 'dataset_list_file' in config.yaml to specify the dataset"

    crop_size = config["preprocessing"]["crop_size"]
    nb_levels = config["model"]["nb_levels"]
    assert (np.all(np.array(crop_size) % (2**(nb_levels-1)) == 0)), f"crop_size {crop_size} needs to be divisible by 2^{nb_levels-1}"
    ndims = config["model"]["ndims"]
    assert (ndims == len(crop_size)), f"crop_size {crop_size} is not for {ndims}D"

    weight_init = config["model"].get("weight_init", None)
    if (weight_init is not None and weight_init not in ['xavier_uniform', 'zeros']):
        mainlogger.error(f"weight_init '{weight_init}' is not supported. The options are either 'xavier_uniform' or 'zeros'")
        return

    # setup and configure root and main logger
    output_folder = os.path.abspath(train_output_folder)    
    if (not os.path.exists(output_folder)):
        os.makedirs(output_folder)
    logfile = args.logfile if (args.logfile) else os.path.join(output_folder, 'freeseg_train.log')
    config_logger(logfile=logfile)

    # print the command
    cwd = os.getcwd()
    cmd = ' '.join(sys.argv)
    mainlogger.info("===================== Current date and time: " + str(datetime.datetime.now()) + " =====================")
    mainlogger.info("CWD: " + cwd)
    mainlogger.info("CMD: " + cmd)

    deterministic = config["training"].get("deterministic", False)
    if (deterministic):
        # ??? todo: for multi-process dataloader, use worker_init_fn() and generator to preserve reproducibility
        #           see https://pytorch.org/docs/stable/notes/randomness.html
        set_deterministic_training()

    # save config and dataset_list_file
    # no config updates should happen after this line
    config_saveas = os.path.join(output_folder, "config.yaml")
    shutil.copyfile(args.config, os.path.join(output_folder, "input_config.yaml"))  # --config <>
    Config.save(config, cwd=cwd, cmd=cmd, saveas=config_saveas)                     # updated with command line args
    dataset_list_saveas = os.path.join(output_folder, "dataset_list.yaml")
    shutil.copyfile(config["dataset"]["dataset_list_file"], dataset_list_saveas)

    # Access updated configuration values
    augmentation_class = config["preprocessing"].get("augmentation_class", "freeseg.augmentation.augmentbase.AugmentBase")
    train_augmentations = remove_duplicates(config["preprocessing"].get("train_augmentations"))
    evaluation_augmentations = remove_duplicates(config["evaluation"].get("evaluation_augmentations"))    
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
    labels_segmentation = sorted(config["dataset"]["expected_classes"])
    label_mapping = {label:i for i, label in enumerate(labels_segmentation)}
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    config["dataset"]["label_mapping"] = label_mapping
    train_dataset, validation_dataset, _ = load_datasets(config, augmentation_class,
                                                         train_augmentations, evaluation_augmentations,
                                                         device=preprocessing_device, check_augment=args.check_augment, keep_trainset_in_memory=args.keep_trainset_in_memory)
    perform_evaluation = config["training"].get("perform_evaluation", False)
    if (perform_evaluation and validation_dataset is None):
        mainlogger.error(f"No 'validation' set in {config['dataset']['dataset_list_file']} to perform evaluation")
        return

    # Create training DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True,
                              pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

    input_shape, unique_classes, label_lookup = train_dataset.preload()

    # output segmentation_labels.npy in training directory
    f_segmentation_labels = os.path.join(output_folder, "segmentation_labels.npy")
    np.save(f_segmentation_labels, np.array(sorted(unique_classes)).astype(int))
    
    # create validation DataLoader
    validation_loader = None
    if (perform_evaluation):
        best_model_metric = config["training"]["best_model_metric"]
        validation_loader = DataLoader(validation_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    mainlogger.info("Training Device: {}".format(device) + (f' (GPU index: {gpu_index})' if (gpu_index is not None) else ''))
    mainlogger.info(f"model: {config['model'].get('name')}")
    if (checkpoint is not None):
        mainlogger.info(f"resume training from model: {checkpoint}")
    elif (weight_init is not None):
        mainlogger.info(f"weight_init: {weight_init}")
    if (config["training"].get("wl2_epochs", 0) > 0):
        mainlogger.info(f"wl2_epochs: {config['training'].get('wl2_epochs')}")
        mainlogger.info(f"wl2_metrics: {config['training'].get('wl2_metrics', 'freeseg.metrics.WeightedL2Loss')}")
    if (config["training"].get("dice_epochs", 0) > 0):
        mainlogger.info(f"dice_epochs: {config['training'].get('dice_epochs')}")
        mainlogger.info(f"model_metrics: {config['training'].get('model_metrics', 'freeseg.metrics.DiceLoss')}")
    mainlogger.info(f"keep_trainset_in_memory: {args.keep_trainset_in_memory}")
    mainlogger.info(f"deterministic: {deterministic}")
    mainlogger.info(f"perform_evaluation: {perform_evaluation}")    
    if (perform_evaluation):
        mainlogger.info(f"best_model_metric: {best_model_metric}")
    mainlogger.info("Preprocessing Device: {}".format(preprocessing_device) + (f' (GPU index: {gpu_index})' if (gpu_index is not None) else ''))
    mainlogger.info(f"Preprocessing augmentation_class: {augmentation_class}")
    mainlogger.info(f"Preprocessing train_augmentations: {train_augmentations}")
    mainlogger.info(f"Preprocessing pin_memory: {pin_memory}")
    mainlogger.info(f"Preprocessing num_workers: {num_workers}")
    mainlogger.info(f"Preprocessing prefetch_factor: {prefetch_factor}")
    mainlogger.info(f"Preprocessing persistent_workers: {persistent_workers}")
    #mainlogger.info(f"Preprocessing check_augment: {args.check_augment}")
    mainlogger.info(f"Preprocessing sampling_hyperparameters: {config['preprocessing'].get('sampling_hyperparameters', True)}")

    mainlogger.info(f"batch_size: {config['training']['batch_size']}")
    mainlogger.info(f"crop_size: {crop_size}")
    mainlogger.info(f"color table: {ctab}")
    mainlogger.info(f"train_output_folder: {output_folder}")        
    mainlogger.info(f"training config: saved as {config_saveas}")
    mainlogger.info(f"dataset list: saved as {dataset_list_saveas}")
    mainlogger.info("")
    if (logfile is not None):
        mainlogger.info(f"training log: {logfile}")

    # save label_mapping/inverse_label_mapping in train_dataset_dict
    train_dataset_dict = {
        "batch_size": config["training"]["batch_size"],
        "segmentation_labels": labels_segmentation,
        "label_mapping": label_mapping,
        "inverse_label_mapping": inverse_label_mapping,
        "crop_size": crop_size,
        "num_samples": len(train_dataset),
        "input_shape": input_shape[1:],
        "num_channels": input_shape[0],
        "priors": train_dataset.haspriors(),
    }

    train(train_loader, config, output_folder, len(unique_classes), ctab,
          label_lookup=label_lookup, checkpoint=checkpoint, validation_loader=validation_loader, device=device, preprocessing_device=preprocessing_device, gpu_index=gpu_index,
          train_dataset_dict=train_dataset_dict, debug=args.debug, weight_init=weight_init)

    # check memory usage
    if (args.vmp):
        print_vm_peak()

    mainlogger.info("Done!")
                       
    
def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--model_name", type=str, help="Class used to create the model to train")
    parser.add_argument("--dataset_list_file", type=str, help="Path to the dataset list file")
    parser.add_argument("--keep_trainset_in_memory", action='store_true', help="Keep preloaded training data in memory")
    parser.add_argument("--deterministic", action='store_true', help="deterministic training")
    parser.add_argument("--ctab", type=str, help="Path to the lookup table")
    parser.add_argument("--train_output_folder", type=str, default=None, help="Folder for saving training outputs")    
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint file to resume training from")
    parser.add_argument("--cpu", action='store_true', help="Run on CPU.")
    parser.add_argument("--num_workers", type=int, help="Number of Dataloader workers")
    parser.add_argument("--prefetch_factor", type=int, help="Number of batches loaded in advance by each worker")
    parser.add_argument("--pin_memory", action='store_true', help="Store data in pinned memory")
    parser.add_argument("--persistent_workers", action='store_true', help=" Keep the workers Dataset instances alive")
    parser.add_argument("--crop_size", nargs="+", type=int, help="Crop size for training and validation")
    parser.add_argument("--batch_size", type=int, help="Batch size for DataLoader")
    #parser.add_argument("--expected_classes", nargs="+", type=int, help="Expected classes in the dataset")
    parser.add_argument("--write_tensorboard_summary", action='store_true', help="Write tensorboard summary")
    parser.add_argument("--perform_evaluation", action='store_true', help="Perform evaluation after each epoch")
    parser.add_argument("--best_model_metric", type=str, default=None, choices=["loss", "dice"], help="Metric for saving the best model (loss or dice)")
    parser.add_argument("--check_augment", action='store_true', help="Reject augmentations not having all the labels")
    parser.add_argument("--weight_init", type=str, help="How to init network weights, 'zeros' or 'xavier_uniform'")
    parser.add_argument('--vmp', action='store_true', help='Enable printing of vmpeak at the end.')
    parser.add_argument('--logfile', type=str, help='Set logfile (default is freeseg_train.log)')
    parser.add_argument("--debug", action='store_true', help="Output volumes for debugging.")
    parser.add_argument("--verbose", action='store_true', help="Print debug info to stdout")

    # parse commandline
    args = parser.parse_args()

    return args


def train(train_loader, config, train_output_folder, num_labels, ctab, label_lookup=None, checkpoint=None,
          validation_loader=None, device=None, preprocessing_device=None, gpu_index=None, train_dataset_dict=None, debug=False, weight_init=None):
    # create the model to train
    model_arch_dict = config["model"]
    model_arch_dict["num_channels"] = config["dataset"]["expected_num_channels"]
    model_arch_dict["nb_labels"] = len(config["dataset"]["expected_classes"])
    model_arch_dict["add_priors"] = train_dataset_dict.get("priors", False)
    if (weight_init is not None):
        model_arch_dict["weight_init"] = weight_init

    the_model_name = model_arch_dict.get("name", None)
    assert the_model_name is not None, "Model name is not available."

    model_class = get_class(the_model_name, "freeseg.models.unet")
    model = model_class(model_arch_dict).to(device)    
   
    # print Model Architecture
    # from torchinfo import summary
    # input_shape = train_dataset_dict["input_shape"]
    # summary(model, input_size=input_shape)

    # create the Training object
    trainer = Training(train_output_folder,
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
                       gpu_index=gpu_index,
                       preprocessing_device=preprocessing_device,
                       debug=debug)
                       
    # train wl2 epochs
    wl2_epochs = config["training"].get("wl2_epochs", 0)
    if (wl2_epochs > 0):
        wl2_metrics = get_class(config["training"].get("wl2_metrics", "freeseg.metrics.WeightedL2Loss"), "freeseg.metrics")
        mainlogger.info(f"training {wl2_epochs} wl2 epochs: {wl2_metrics} ...")
        wl2_loss_fn = wl2_metrics()
        trainer.train_model(lr=config["training"]["pre_train_learning_rate"],
                            epochs=wl2_epochs,
                            steps_per_epoch=config["training"]["steps_per_epoch"],
                            metric_type='wl2',
                            loss_fn=wl2_loss_fn)
                       
    # train dice epochs
    dice_epochs = config["training"].get("dice_epochs", 0)
    if (dice_epochs > 0):
        model_metrics = get_class(config["training"].get("model_metrics", "freeseg.metrics.DiceLoss"), "freeseg.metrics")
        mainlogger.info(f"training {dice_epochs} epochs: {model_metrics} ...")
        dice_loss_fn = model_metrics(
            num_classes=num_labels,
            dice_type="soft"
        )                   
        trainer.train_model(lr=config["training"]["pre_train_learning_rate"],
                            epochs=dice_epochs,
                            steps_per_epoch=config["training"]["steps_per_epoch"],
                            metric_type='dice',
                            loss_fn=dice_loss_fn)                   


# execute script
if __name__ == '__main__':
    main()
