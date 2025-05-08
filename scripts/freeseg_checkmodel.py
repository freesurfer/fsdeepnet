#!/usr/bin/env python

import os
import logging
import sys
import torch
import argparse

from freeseg import models
from freeseg.config import Config
from freeseg.checkpoint import Checkpoint
from freeseg.utils import get_class

description = """
Usage: check_model.py 
       --config <config.yaml> | --checkpoint <checkpoint> [--weight_outdir <weight_outdir>]
       [--ndims <n>]
       [--input_shape <H W (D)>]
       [--cpu]

       1. Use '--config <>' to create the network, or 
              '--checkpoint <> to load a pre-trained network.
       2. Use '--weight_outdir <>' to save trainable parameters in <checkpoint>.
          The option is ignored if it is not run with '--checkpoint <>'.
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

    if (args.config is not None and args.checkpoint is not None):
        logging.error("'--config <>' and '--checkpoint <>' are mutually exclusive")
        sys.exit(1)
    if (args.config is None and args.checkpoint is None):
        logging.error("Use '--config <>' to create the network, or '--checkpoint <>' to load a pre-trained network")
        sys.exit(1)

    if (args.cpu):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_channels = 1
    if (args.config is not None):
        # Load config file
        config = Config.load(args.config)

        # create the model
        model_arch_dict = config["model"]
        model_arch_dict["num_channels"] = config["dataset"]["expected_num_channels"]
        model_arch_dict["nb_labels"] = len(config["dataset"]["expected_classes"])
        #model_arch_dict["add_priors"] = train_dataset_dict.get("priors", False)
        
        the_model_name = model_arch_dict.get("name", None)
        assert the_model_name is not None, "Model name is not available."
        n_channels = model_arch_dict["num_channels"]
    else:
        # Load the Trained Model
        if not os.path.isfile(args.checkpoint):
            logging.error('ERROR: file does not exist: %s' % args.checkpoint)
            sys.exit(1)

        checkpoint = Checkpoint()
        checkpoint.load(args.checkpoint, device=device)
        assert checkpoint.model_arch_dict is not None, "Model architecture information not available."
        assert checkpoint.train_dataset_dict is not None, "Training dataset information not available."

        the_model_name = checkpoint.model_arch_dict.get("name", None)
        assert the_model_name is not None, "Model name is not available."

        model_arch_dict = checkpoint.model_arch_dict
        n_channels = checkpoint.train_dataset_dict["num_channels"]

    model_class = get_class(the_model_name, "freeseg.models.unet")
    model = model_class(model_arch_dict).to(device)
    if (args.checkpoint is not None):
        model.load_state_dict(checkpoint.model_state_dict)
        if (args.weight_outdir is not None):
            # output model trainable parameters in given checkpoint
            import numpy as np
            if (not os.path.exists(args.weight_outdir)):
                os.makedirs(args.weight_outdir)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    path_npy = f"{args.weight_outdir}/{name}.npy"
                    np.save(path_npy, param.data.cpu().numpy())

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
    parser.add_argument("--weight_outdir", type=str, help="Directory to save trainable parameters in given checkpoint")
    parser.add_argument("--ndims", type=int, default=3, help="Number of image dimensions, 2D or 3D")
    parser.add_argument("--input_shape", nargs="+", type=int, default=(160,160,160), help="Network image input shape, ex. 160 160 160")
    parser.add_argument("--cpu", action='store_true', help="Run on CPU")

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    # parse commandline
    args = parser.parse_args()

    return args


# execute script
if __name__ == '__main__':
    main()
