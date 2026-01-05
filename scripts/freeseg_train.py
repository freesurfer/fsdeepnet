#!/usr/bin/env python

import sys
import logging
import argparse

from freeseg import models
from freeseg.training import Training
from freeseg.config import Config
from freeseg.utils import utility as utils

"""
Usage: freeseg_train.py 
       --config <config.yaml>
       [--train_output_folder <train_output_folder>]
       [--keep_trainset_in_memory]
       [--deterministic]
       [--model_name <model_classname>]
       [--dataset_list_file <dataset_list_file>  --train_cohort <train|validation|test> --validation_cohort <train|validation|test>]
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

    config = Config.process(args, logger=mainlogger)
    config, train_loader, validation_loader, _, model, optimizer_cls, _ = Training.setup(config, preload_dataset=args.preload)
    Config.print(config, mainlogger)

    train(config, train_loader, model, optimizer_cls,
          validation_loader=validation_loader)

    # check memory usage
    if (config["vmp"]):
        utils.print_vm_peak()

    mainlogger.info("Done!")
                       
    
def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--model_name", type=str, help="Class used to create the model to train")
    parser.add_argument("--dataset_list_file", type=str, help="Path to the dataset list file")
    parser.add_argument("--res_diff_thresh", type=float, help="Training data resolution diff. threshold (percentage in its decimal form)")
    parser.add_argument("--keep_trainset_in_memory", action='store_true', help="Keep preloaded training data in memory")
    parser.add_argument("--preload", action='store_true', help="Preload training dataset")
    parser.add_argument("--train_cohort", nargs="+", type=str, default=['train'], help="Specify training dataset cohort. Can be combinations of train, validation, or test")
    parser.add_argument("--validation_cohort", nargs="+", type=str, default=['validation'], help="Specify validation dataset cohort. Can be combinations of train, validation, or test")
    parser.add_argument("--deterministic", action='store_true', help="deterministic training")
    parser.add_argument("--ctab", type=str, help="Path to the lookup table")
    parser.add_argument("--train_output_folder", type=str, default=None, help="Folder for saving training outputs")
    parser.add_argument("--report_moving_avg", action='store_true', help="Report simple moving loss and dice average for each training step.")    
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint file to resume training from")
    parser.add_argument("--cpu", action='store_true', help="Run on CPU.")
    parser.add_argument("--num_workers", type=int, help="Number of Dataloader workers")
    parser.add_argument("--prefetch_factor", type=int, help="Number of batches loaded in advance by each worker")
    parser.add_argument("--pin_memory", action='store_true', help="Store data in pinned memory")
    parser.add_argument("--persistent_workers", action='store_true', help="Keep the workers Dataset instances alive")
    parser.add_argument("--crop_size", nargs="+", type=int, help="Crop size for training and validation")
    parser.add_argument("--batch_size", type=int, help="Batch size for DataLoader")
    parser.add_argument("--wl2_epochs", type=int, help="Number of wl2 training epochs")
    parser.add_argument("--dice_epochs", type=int, help="Number of dice training epochs")
    parser.add_argument("--learning_rate", type=float, help="Network learning rate")
    parser.add_argument("--nb_labels", type=int, help="Number of segmentation labels")
    parser.add_argument("--nb_levels", type=int, help="Number of network levels")
    parser.add_argument("--nb_features", type=int, help="Number of features at the first level")
    parser.add_argument("--feat_mult", type=int, help="Feature multiplication factor")
    parser.add_argument("--nb_conv_per_level", type=int, help="Number of convolution layers at each level")
    parser.add_argument("--conv_size", type=int, help="Convolution kernel size")
    parser.add_argument("--pool_size", type=int, help="Max pooling size")
    parser.add_argument("--use_residuals", action='store_true', help="Use residuals")
    #parser.add_argument("--expected_classes", nargs="+", type=int, help="Expected classes in the dataset")
    parser.add_argument("--write_tensorboard_summary", action='store_true', help="Write tensorboard summary")
    parser.add_argument("--perform_evaluation", action='store_true', help="Perform evaluation after each epoch")
    parser.add_argument("--best_model_metric", type=str, default=None, choices=["loss", "dice"], help="Metric for saving the best model (loss or dice)")
    parser.add_argument("--weight_init", type=str, help="How to init network weights, 'zeros' or 'xavier_uniform'")
    parser.add_argument('--vmp', action='store_true', help='Enable printing of vmpeak at the end.')
    parser.add_argument('--logfile', type=str, help='Set logfile (default is freeseg_train.log)')
    parser.add_argument("--debug", action='store_true', help="Output volumes for debugging.")
    parser.add_argument("--verbose", action='store_true', help="Print debug info to stdout")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    # parse commandline
    args = parser.parse_args()

    return args


def train(config, train_loader, model, optimizer_cls, validation_loader=None):
    # print model_arch_dict
    model_arch_dict = model.arch_dict
    models.model_arch(model_arch_dict, logger=mainlogger)

    if (config["verbose"]):
        # print model summary and trainable parameters
        net_crop_size = config["preprocessing"]["crop_size"]
        net_input_shape = (model_arch_dict["num_channels"], *net_crop_size)
        models.model_summary(model, net_input_shape, logger=mainlogger)
        models.model_parameters(model, logger=mainlogger)

    checkpoint = config["model_checkpoint"]
    if (checkpoint is not None):
        mainlogger.info(f"Resuming training from checkpoint: {checkpoint}")

    # retrieve data_generator information
    cfg_data_generator = config["training"].pop("data_generator", {})
    if (cfg_data_generator):
        data_generator = utils.get_class(cfg_data_generator.pop("fn", "freeseg.utils.utility.DataGenerator"))
    else:
        data_generator = utils.get_class("freeseg.utils.utility.DataGenerator")
    cfg_data_generator.update({"device" : config["preprocessing_device"]})
    fn_data_generator = data_generator(train_loader, **cfg_data_generator)

    # retrieve model_metrics_accuracy
    model_metrics_accuracy = config["training"].pop("model_metrics_accuracy", None)
    dice_hard_fn = None
    if (model_metrics_accuracy is not None):
        cls_model_metrics_accuracy = utils.get_class(model_metrics_accuracy.pop("class_name", "freeseg.metrics.DiceDice"))
        # '**' operator unpacks rest of dictionary key/value pairs to keyword arguments
        dice_hard_fn = cls_model_metrics_accuracy(**model_metrics_accuracy)

    # retrieve wl2/dice epochs training information
    wl2_epochs = config["training"].pop("wl2_epochs", 0)
    wl2_metrics = config["training"].pop("wl2_metrics", None)
    dice_epochs = config["training"].pop("dice_epochs", 0)
    model_metrics = config["training"].pop("model_metrics", None)

    # create the Training object
    trainer_cls = utils.get_class(config["training"].get("trainer_class", "freeseg.training.Training"))        
    trainer = trainer_cls(dnn=model,
                          train_loader=train_loader,
                          fn_data_generator=fn_data_generator,
                          model_arch_dict=model_arch_dict,
                          train_dataset_dict=config["dataset"],
                          validation_loader=validation_loader,
                          accuracy_fn=dice_hard_fn,
                          ctab=config["ctab"],
                          model_checkpoint=config["model_checkpoint"],
                          device=config["device"],
                          gpu_index=config["gpu_index"],
                          debug=config["debug"],
                          **config["training"],
                          #train_output_folder=config["training"]["train_output_folder"],                          
                          #best_model_metric=config["training"]["best_model_metric"],
                          #write_tensorboard_summary=config["training"].get("write_tensorboard_summary", False),
                          #report_moving_avg=config["training"].get("report_moving_avg", False),
                         )

    # train wl2 epochs
    if (wl2_epochs > 0 and wl2_metrics is not None):
        cls_wl2_metrics = utils.get_class(wl2_metrics.pop("class_name", "freeseg.metrics.WeightedL2Loss"))
        if (checkpoint is None):
            mainlogger.info(f"training {wl2_epochs} wl2 epochs: {trainer_cls}, {optimizer_cls}, {cls_wl2_metrics}, lr:{config['training']['pre_train_learning_rate']} ...")
        # '**' operator unpacks rest of dictionary key/value pairs to keyword arguments
        wl2_loss_fn = cls_wl2_metrics(**wl2_metrics)
        trainer.train_model(lr=config["training"]["pre_train_learning_rate"],
                            epochs=wl2_epochs,
                            steps_per_epoch=config["training"]["steps_per_epoch"],
                            metric_type='wl2',
                            optimizer_cls=optimizer_cls,
                            loss_fn=wl2_loss_fn)

    # train dice epochs
    if (dice_epochs > 0 and model_metrics is not None):
        cls_model_metrics = utils.get_class(model_metrics.pop("class_name", "freeseg.metrics.DiceLoss"))
        mainlogger.info(f"training {dice_epochs} dice epochs: {trainer_cls}, {optimizer_cls}, {cls_model_metrics}, lr:{config['training']['learning_rate']} ...")
        # '**' operator unpacks dictionary key/value pairs to keyword arguments        
        dice_loss_fn = cls_model_metrics(**model_metrics)
        trainer.train_model(lr=config["training"]["learning_rate"],
                            epochs=dice_epochs,
                            steps_per_epoch=config["training"]["steps_per_epoch"],
                            metric_type='dice',
                            optimizer_cls=optimizer_cls,
                            loss_fn=dice_loss_fn)


# execute script
if __name__ == '__main__':
    main()
