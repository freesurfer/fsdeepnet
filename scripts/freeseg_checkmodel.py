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
                                                            [--weights <model_state_key>:weight_outdir>]
                                                            [--saveas <> 
                                                              [--strip <dictkey1 dictkey2 ...>]
                                                              [--rename <fromkey1:tokey1 fromkey2:tokey2 ...>]
                                                              [--update label_lookup:<> segmentation_names:<> topology_classes:<> target_res:<> model_class:<> config:config]
                                                              [--prefix-model_layer <prefix>]
                                                              [--replace-model_layer <from:to>] ] ]
       [--ndims <n>]
       [--input_shape <H W (D)>]
       [--cpu]

       1. Use '--config <>' to create the network, or 
              '--checkpoint <> to load a pre-trained network.
       2. Use '--weights <model_state_key>:weight_outdir>' to save trainable parameters in <checkpoint>.
          The option is ignored if it is not run with '--checkpoint <>'.
       3. Options to update pre-trained model: 
          a) '--strip <dictkey1 dictkey2 ...>': strip <dictkey> from checkpoint
             <dictkey> needs to match dict keywords in checkpoint
          b) '--update label_lookup:<> segmentation_names:<> topology_classes:<> target_res:<> model_class:<> config:config'
             update pre-trained model with given key:value pairs
             if key:value=='config:config', both train_dataset_dict and model_arch_dict are updated using the config.yaml
          c) '--prefix-model_layer <prefix>'
             it is assumed the model state is saved under keyword 'model_state_dict'
             <prefix> = the prefix to be added to each model state layer name
          d) '--replace-model_layer <from:to>'
             it is assumed the model state is saved under keyword 'model_state_dict'
             replace <from> with <to> for each model state layer name
          e) '--rename <fromkey1:tokey1 fromkey2:tokey2 ...>'
             rename the checkpoint top level dictionary keywords

Example 1: retrieve checkpoint top level information
       fspython freeseg_checkmodel.py --checkpoint orig.pth --info

Example 2: retrieve information of 30 layers of model_state_dict
       fspython freeseg_checkmodel.py --checkpoint orig.pth --info --detail --keys model_state_dict --nkey 30

Example 3: rename 'model_state' to 'model_state_dict'
       fspython freeseg_checkmodel.py --checkpoint orig.pth --rename model_state:model_state_dict --saveas new.pth

Example 4: strip 'optimizer_state'
       fspython freeseg_checkmodel.py --checkpoint orig.pth --strip optimizer_state --saveas new.pth

Example 5: prefix model state layer
       fspython freeseg_checkmodel.py --checkpoint orig.pth --prefix-model_layer unet3d. --saveas new.pth

Example 6: replace model state layer
       fspython freeseg_checkmodel.py --checkpoint orig.pth --replace-model_layer unet3d:unet --saveas new.pth

Example 7: update checkpoint
       fspython freeseg_checkmodel.py --checkpoint orig.pth
         --update topology_classes:topological_classes.npy segmentation_names:segmentation_names.npy
         --saveas new.pth

Example 8: update checkpoint 'train_dataset_dict' and 'model_arch_dict' using information from config.yaml
       fspython freeseg_checkmodel.py --checkpoint orig.pth --update config:config --saveas new.pth
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

    info = True if (args.info) else False
    update = False
    if (args.rename is not None or \
        args.strip is not None or \
        args.update is not None or \
        args.prefix_model_layer is not None or \
        args.replace_model_layer is not None):
      update = True

    # check command line options
    if (args.config is None and args.checkpoint is None):
        logging.error("[ERROR] Use '--config <>' to create the network, or '--checkpoint <>' to load a pre-trained network")
        sys.exit(1)
    if (info or update):
        assert (args.checkpoint is not None), f"Use '--checkpoint <>' to load a pre-trained network"
    if (args.update is not None and "config:config" in args.update):
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
            logging.error('[ERROR] file does not exist: %s' % args.checkpoint)
            sys.exit(1)

        checkpoint = Checkpoint()
        checkpoint.load(args.checkpoint, device=device)

        if (checkpoint.model_arch_dict is None):
            logging.warning("[WARN] Model architecture information not available")
            info = True

    # '--weights <model_state_key>:weight_outdir'
    if (args.weights is not None):
        if (args.checkpoint is None):
            logging.warning(f"[WARN] 'No checkpoint specified, --weights {args.weights} ignored.")
        else:
            model_state_key, weight_outdir = args.weights.split(':')
            # retrieve model state using the given key 'model_state_key'. the key can be different in different checkpoint.
            model_state = checkpoint.dict.get(model_state_key, None)
            if (model_state is None):
                logging.error(f"[ERROR] dict key '{model_state_key}' doesn't exist")
                sys.exit(1)
        
            # output model trainable parameters in given checkpoint
            import numpy as np
            if (not os.path.exists(weight_outdir)):
                os.makedirs(weight_outdir)
            logging.info(f"weight/bias can be found in directory {weight_outdir}")
            for k, v in model_state.items():
                path_npy = f"{weight_outdir}/{k}.npy"
                np.save(path_npy, v.cpu().numpy())

    # '--info' or input checkpoint has no model_arch_dict information
    if (info):
        Checkpoint.print(checkpoint.dict, detail=args.detail, nkeys=args.nkeys, keys=args.keys, report_type=args.report_type)

    # update checkpoint
    if (update):
        update_checkpoint(checkpoint, args.saveas, config, args.update, args.strip, args.rename, args.prefix_model_layer, args.replace_model_layer)

    if (info or update or checkpoint.model_arch_dict is None):
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
        
    model_class = utils.get_class(the_model_name)
    model = model_class(model_arch_dict).to(device)

    # print network summary
    models.model_arch(model_arch_dict)
    models.model_print(model)
    models.model_summary(model, (n_channels, *args.input_shape[-args.ndims:]), device=device, debug=True)
    #models.model_parameters(model)
    #models.model_summary_torchinfo(model, (1, 1, *args.input_shape[-args.ndims:]))
        

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
    parser.add_argument("--weights", type=str, help="Save trainable parameters in given checkpoint, <model_state_key:weight_outdir>")
    parser.add_argument("--strip", nargs="+", type=str, help="Strip dict keys from checkpoint")
    parser.add_argument("--rename", nargs="+", type=str, help="Rename checkpoint top level dictionary keywords, <fromkey1:tokey1 fromkey2:tokey2 ...>")
    parser.add_argument("--prefix-model_layer", type=str, help="Prepend 'prefix' to each model state layer name")
    parser.add_argument("--replace-model_layer", type=str, help="Replace model state layer names, <from:to>")
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


def update_checkpoint(checkpoint, saveas, config, toupdates, stripkeys, renamekeys, layerprefix, layerreplace):
    # '--strip <dictkeys>'
    if (stripkeys is not None):
        checkpoint.strip(stripkeys)

    # '--rename <fromkey1:tokey1 fromkey2:tokey2 ...>'
    if (renamekeys is not None):
        checkpoint.rename(renamekeys)

    # '--prefix-model_layer <prefix>'
    if (layerprefix is not None):
        checkpoint.prefix_model_layer(layerprefix)

    # '--replace-model_layer <from:to>'
    if (layerreplace is not None):
        checkpoint.replace_model_layer(layerreplace)

    # '--update label_lookup:<> segmentation_names:<> topology_classes:<> target_res:<> model_class:<> config:config'
    if (toupdates is None):
        toupdates = {}  # empty dict
    dict_update = dict()    
    for toupdate in (toupdates):
        (key, value) = toupdate.split(":")
        logging.info(f"update {key}:{value}")

        if (key == "label_lookup"):
            import surfa as sf
            label_lookup = sf.load_label_lookup(value)
            dict_update.update({"label_lookup" : value})
        #    
        elif (key == "segmentation_names"):
            checkpoint.train_dataset_dict["segmentation_names"] = value
            dict_update.update({"train_dataset_dict" : checkpoint.train_dataset_dict})
        #
        elif (key == "topology_classes"):
            checkpoint.train_dataset_dict["topology_classes"] = value
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

    logging.info(f"save updated checkpoint as {saveas}")
    checkpoint.save(saveas, dict_update)

        
# execute script
if __name__ == '__main__':
    main()
