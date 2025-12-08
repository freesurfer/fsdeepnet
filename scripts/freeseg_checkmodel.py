#!/usr/bin/env python

import os
import logging
import sys
import numpy
import torch
import argparse

from freeseg import models
from freeseg.config import Config
from freeseg.training import Training
from freeseg.checkpoint import Checkpoint
from freeseg.utils import utility as utils

description = """
Usage: freeseg_checkmodel.py 
       --config <config.yaml> | --checkpoint <checkpoint> [ [--info [--detail] [--report_type] [--nkeys <n>] [--keys <>] ]
                                                            [--weight_outdir <weight_outdir>]
                                                            [--saveas <> 
                                                              [--strip_optimizer_state]
                                                              [--update label_lookup:<> segmentation_names:<> topology_classes:<> target_res:<> model_class:<> config:config] ] ]
       [--ndims <n>]
       [--input_shape <H W (D)>]
       [--cpu]

       1. Use '--config <>' to create the network, or 
              '--checkpoint <> to load a pre-trained network.
       2. Use '--weight_outdir <>' to save trainable parameters in <checkpoint>.
          The option is ignored if it is not run with '--checkpoint <>'.
       3. Options to update pre-trained model: 
          a) '--strip_optimizer_state': strip optimizer states
          b) '--update label_lookup:<> segmentation_names:<> topology_classes:<> target_res:<> model_class:<> config:config'
             update pre-trained model with given key:value pairs
             if key:value=='config:config', both train_dataset_dict and model_arch_dict are updated using the config.yaml

Example:
       fspython freeseg_checkmodel.py --checkpoint 7233351_dice_300.pth --strip_optimizer_state --saveas new2.7233351_dice_300.pth 
       --update topology_classes:/usr/local/freesurfer/8.0.0/models/synthseg_topological_classes_2.0.npy segmentation_names:/autofs/space/azura_003/users/synthseg-training/synthseg/numpy_vectors/segmentation_names_lut_2.0.npy
"""

# Configure logging settings
logging.basicConfig(
    level=logging.DEBUG,  # Set the log level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(message)s",   #"%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to terminal
    ],
)

def main():
    args = argument_parse(description)

    update = True if (args.strip_optimizer_state or args.update is not None) else False
    info = True if (args.info) else False

    # check command line options
    if (args.config is None and args.checkpoint is None):
        logging.error("Use '--config <>' to create the network, or '--checkpoint <>' to load a pre-trained network")
        sys.exit(1)
    if (args.info or args.strip_optimizer_state or args.update is not None or args.weight_outdir is not None):
        assert (args.checkpoint is not None), f"Use '--checkpoint <>' to load a pre-trained network"
    if (args.update is not None and args.update == "config"):
        assert (args.config is not None), f"Specify '--config <>' for '--update {args.update}'"
    if (update):
        assert (args.saveas is not None), f"Specify path to save new checkpoint as '--saveas <>'"

    if (args.cpu):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config file
    config = None
    if (args.config is not None):
        config = Config.process(args, logger=logging, require_train_outfolder=False, require_dataset_list=False)
        config, _, _, _, _, _ = Training.setup(config, preload_dataset=False, create_train_dataset=False, create_loader=False, create_model=False)

    # load pre-trained model
    checkpoint = None
    if (args.checkpoint is not None):
        if not os.path.isfile(args.checkpoint):
            logging.error('ERROR: file does not exist: %s' % args.checkpoint)
            sys.exit(1)

        checkpoint = Checkpoint()
        checkpoint.load(args.checkpoint, device=device)

        if (checkpoint.model_arch_dict is None):
            info = True

    # '--info' or input checkpoint has no model_arch_dict information
    if (info):
        if (checkpoint.model_arch_dict is None):
            print("WARN: Model architecture information not available")
        Checkpoint.print(checkpoint.dict, detail=args.detail, nkeys=args.nkeys, keys=args.keys, report_type=args.report_type)
        sys.exit(0)

    # update checkpoint
    if (update):
        update_checkpoint(args.update, config, checkpoint, args.saveas, args.strip_optimizer_state)
        sys.exit(0)

    # create the network from config or re-construct it from pre-trained model
    model_arch_dict = None
    if (checkpoint is not None):
        model_arch_dict = checkpoint.model_arch_dict
    else:
        model_arch_dict = config["model"]

    assert (model_arch_dict is not None), "Model architecture information not available."        
    the_model_name = model_arch_dict.get("name", None)
    assert the_model_name is not None, "Model class is not available."
    n_channels = model_arch_dict.get("num_channels", 1)
        
    model_class = utils.get_class(the_model_name, "freeseg.models.unet")
    model = model_class(model_arch_dict).to(device)

    # print network summary                    
    models.model_arch(model_arch_dict)
    models.model_print(model)
    models.model_summary(model, (n_channels, *args.input_shape[-args.ndims:]), device=device, debug=True)
    #models.model_parameters(model)
    #models.model_summary_torchinfo(model, (1, 1, *args.input_shape[-args.ndims:]))

    if (args.weight_outdir is not None):
        model.load_state_dict(checkpoint.model_state_dict)
        
        # output model trainable parameters in given checkpoint
        import numpy as np
        if (not os.path.exists(args.weight_outdir)):
            os.makedirs(args.weight_outdir)
        logging.info(f"weight/bias can be found in directory {args.weight_outdir}")
        for name, param in model.named_parameters():
            if param.requires_grad:
                path_npy = f"{args.weight_outdir}/{name}.npy"
                np.save(path_npy, param.data.cpu().numpy())
        

def argument_parse(description):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=description)

    # input/outputs
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint file")
    parser.add_argument("--info", action='store_true', help="Report checkpoint root level contents")
    parser.add_argument("--detail", action='store_true', help="Recursively report checkpoint contents in detail, only works with '--info'")
    parser.add_argument("--report_type", action='store_true', help="Report types for instances other than 'dict', 'torch.Tensor', and 'numpy.ndarray', only works with '--detail'")
    parser.add_argument("--nkeys", type=int, default=30, help="Limit the number of dict keys to report, only works with '--detail'")
    parser.add_argument("--keys", nargs="+", type=str, help="List of dict keys to report, only works with '--detail'")
    parser.add_argument("--weight_outdir", type=str, help="Directory to save trainable parameters in given checkpoint")
    parser.add_argument("--strip_optimizer_state", action='store_true', help="Strip optimizer state from pre-trained checkpoint")
    parser.add_argument("--update", nargs="+", type=str, help="Update pre-trained checkpoint dict, choices are 'label_lookup:<>', 'segmentation_names:<>', 'topology_classes:<>', 'model_class:<>', and 'config:config'")
    parser.add_argument("--saveas", type=str, help="Path to save new checkpoint as")
    parser.add_argument("--ndims", type=int, default=3, help="Number of image dimensions, 2D or 3D")
    parser.add_argument("--input_shape", nargs="+", type=int, default=(160,160,160), help="Network image input shape, ex. 160 160 160")
    parser.add_argument("--cpu", action='store_true', help="Run on CPU")

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    # parse commandline
    args = parser.parse_args()

    return args


def update_checkpoint(toupdates, config, checkpoint, saveas, strip_optimizer_state=False):
    dict_update = dict()

    # update checkpoint dict
    for toupdate in (toupdates):
        (key, value) = toupdate.split(":")
        logging.info(f"update {key}:{value}")

        if (key == "label_lookup"):
            import surfa as sf
            label_lookup = sf.load_label_lookup(value)
            #dict_update.update({"label_lookup" : label_lookup})
            dict_update.update({"label_lookup" : value})
        #    
        elif (key == "segmentation_names"):
            checkpoint.train_dataset_dict["segmentation_names"] = value  #np.load(value)
            dict_update.update({"train_dataset_dict" : checkpoint.train_dataset_dict})
        #
        elif (key == "topology_classes"):
            checkpoint.train_dataset_dict["topology_classes"] = value  #np.load(value)
            dict_update.update({"train_dataset_dict" : checkpoint.train_dataset_dict})
        #
        elif (key == "target_res"):
            checkpoint.train_dataset_dict["target_res"] = float(value)
            dict_update.update({"train_dataset_dict" : checkpoint.train_dataset_dict})
        #
        elif (key == "model_class"):
            checkpoint.model_arch_dict["name"] = value
            dict_update.update({"model_arch_dict" : checkpoint.model_arch_dict})
        #
        elif (key == "config"):
            logging.info("update model_arch_dict and train_dataset_dict")
            dict_update.update({"model_arch_dict" : config["model"],
                                "train_dataset_dict" : config["dataset"]})
        #

    # '--strip_optimizer_state'
    if (strip_optimizer_state):
        logging.info("strip optimizer state")
        dict_update.update({"optimizer_state_dict": None})

    logging.info(f"save updated checkpoint as {saveas}")
    checkpoint.save(saveas, dict_update)

        
# execute script
if __name__ == '__main__':
    main()
