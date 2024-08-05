import os
import torch
import torch.optim as optim
import logging
import argparse
import datetime
import numpy as np
import shutil

from torch.utils.data import DataLoader
from models.model import UNet
from training import Training

from utils.train_utils import load_checkpoint
from utils.data_utils import load_config, save_label_mapping, remap_labels
from utils.dataset import load_datasets, dataGenerator
from utils.metrics import WeightedL2Loss, DiceLoss, DiceScore

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
"""

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,  # Set the log level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to terminal
    ],
)

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
    parser.add_argument("--crop_size", nargs="+", type=int, help="Crop size for training and validation")
    #parser.add_argument("--expected_classes", nargs="+", type=int, help="Expected classes in the dataset")
    parser.add_argument("--write_tensorboard_summary", action='store_true', help="Write tensorboard summary")
    parser.add_argument("--perform_evaluation", action='store_true', help="Perform evaluation after each epoch")
    parser.add_argument("--best_model_metric", type=str, default=None, choices=["loss", "dice"], help="Metric for saving the best model (loss or dice)")

    # parse commandline
    args = parser.parse_args()

    return args


# ??? todo: remove label_mapping
def train(input_generator, config, train_output_folder, label_mapping, ctab, input_shape, checkpoint=None, validation_loader=None, device=None):
    # create the model to train
    model_arch_dict = config["model"]
    model_arch_dict["input_shape"] = (config["dataset"]["expected_num_channels"], *input_shape)
    model_arch_dict["nb_labels"] = len(config["dataset"]["expected_classes"])
    model_arch_dict["final_pred_activation"] = config["model"].get("final_pred_activation", "softmax")
    model = UNet(
        input_shape=model_arch_dict["input_shape"],
        ndims=model_arch_dict["ndims"],
        conv_size=model_arch_dict["conv_size"],
        pool_size=model_arch_dict["pool_size"],
        refine_conv=model_arch_dict["refine_conv"],
        nb_features=model_arch_dict["nb_features"],
        nb_levels=model_arch_dict["nb_levels"],
        nb_labels=model_arch_dict["nb_labels"],
        feat_mult=model_arch_dict["feat_mult"],
        nb_conv_per_level=model_arch_dict["nb_conv_per_level"],
        use_residuals=model_arch_dict["use_residuals"],
        use_batchnorm=model_arch_dict["use_batchnorm"],
        activation=model_arch_dict["activation"],
        final_pred_activation=model_arch_dict["final_pred_activation"]).to(device)     
    """
    model = UNet(
        input_shape=(config["dataset"]["expected_num_channels"], *input_shape),
        ndims=config["model"]["ndims"],
        conv_size=config["model"]["conv_size"],
        pool_size=config["model"]["pool_size"],
        refine_conv=config["model"]["refine_conv"],
        nb_features=config["model"]["nb_features"],
        nb_levels=config["model"]["nb_levels"],
        nb_labels=len(config["dataset"]["expected_classes"]),
        feat_mult=config["model"]["feat_mult"],
        nb_conv_per_level=config["model"]["nb_conv_per_level"],
        use_residuals=config["model"]["use_residuals"],
        use_batchnorm=config["model"]["use_batchnorm"],
        activation=config["model"]["activation"],
        final_pred_activation=config["model"].get("final_pred_activation", "softmax")).to(device)
    """
   
    # print Model Architecture
    # from torchinfo import summary
    # summary(model, input_size=input_shape)

    # create the Training object
    trainer = Training(train_output_folder,
                       np.array(config["dataset"]["expected_classes"]),
                       label_mapping,
                       input_generator,
                       model,
                       model_arch_dict=model_arch_dict,
                       ctab=ctab,
                       checkpoint=checkpoint,
                       validation_loader=validation_loader,
                       best_model_metric=config["training"]["best_model_metric"],
                       write_tensorboard_summary=config["training"].get("write_tensorboard_summary", False),
                       device=device)
                       
    # train wl2 epochs (??? todo: make this optional ???)
    wl2_loss_fn = WeightedL2Loss()
    trainer.train_model(lr=config["training"]["pre_train_learning_rate"],
                        epochs=config["training"]["wl2_epochs"],
                        steps_per_epoch=config["training"]["steps_per_epoch"],
                        metric_type='wl2',
                        loss_fn=wl2_loss_fn)
                       
    # train dice epochs
    dice_loss_fn = DiceLoss(
        num_classes=len(label_mapping),
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
    args = argument_parse()
    
    if (args.cpu):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    crop_size = config["preprocessing"]["crop_size"]
    
    # create training/validation dataset with the desired augmentations specified
    train_dataset, validation_dataset, _ = load_datasets(
        config, config["preprocessing"].get("train_augmentations"), config["evaluation"].get("evaluation_augmentations")
    )

    # Create training DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    # ??? todo: we probably can get rid of sample_input_shape too
    sample_input_shape, unique_classes, all_labels = train_dataset.preload()
    input_shape = sample_input_shape[1:]

    # ??? we need label_mapping for dataGenerator
    # ??? move this to Training class, output from labels_segmentation
    label_mapping = save_label_mapping(all_labels, output_folder=output_folder)

    expected_num_channels = config["dataset"]["expected_num_channels"]
    expected_classes = config["dataset"]["expected_classes"]
    assert (
        sorted(unique_classes) == expected_classes
    ), f"Expected classes {expected_classes}, but got {sorted(unique_classes)}"
    assert (
        sample_input_shape[0] == expected_num_channels
    ), f"Expected {expected_num_channels} channels, but got {sample_input_shape[0]}"

    # create validation DataLoader
    validation_loader = None
    perform_evaluation = config["training"].get("perform_evaluation", False)
    if (perform_evaluation):
        best_model_metric = config["training"]["best_model_metric"]
        validation_loader = DataLoader(validation_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    logging.info("Device: {}".format(device))
    if (checkpoint is not None):
        logging.info(f"resume training from model: {checkpoint}")
    logging.info(f"train_root_folder: {train_root_folder}")
    logging.info(f"run_name: {run_name}")
    logging.info(f"crop_size: {crop_size}")
    if (ctab is not None):
        logging.info(f"color table: {ctab}")
    if (perform_evaluation):
        logging.info(f"best_model_metric: {best_model_metric}")
    logging.info(f"training config: saved as {output_folder}/config.yaml")
    logging.info(f"dataset list: saved as {output_folder}/dataset_list.yaml")
    logging.info("Dataset information:")
    logging.info(f"Number of samples in training dataset: {len(train_dataset)}")
    logging.info(f"Number of unique classes: {len(label_mapping)}")
    logging.info(f"Unique class values: {sorted(unique_classes)}")
    logging.info(f"Input shape: {input_shape}")
    logging.info(f"Number of channels: {sample_input_shape[0]}")

    input_generator = dataGenerator(train_loader, device, label_mapping)
                       
    train(input_generator, config, output_folder, label_mapping, ctab, input_shape, checkpoint, validation_loader, device)
                       
