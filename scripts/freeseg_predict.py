#!/usr/bin/env python

import os
import logging
import sys
import numpy as np
import torch
import argparse
import yaml

from freeseg.config import Config
from freeseg.utils import utility as utils
from freeseg.prediction import Prediction


description = """
Usage: freeseg_predict.py 
       [--i <image_path> | --dataset_list_file <dataset.yaml> --cohort <train|validation|test>]
       --o  <output_segmentations>
       --checkpoint <checkpoint>
       [--crop_size <W H D>]
       [--ctab <ctab>]
       [--label <input_labels>]
       [--prior <input_priors>]
       [--gt <ground_truth_dir>] 
       [--noaddctab]
       [--write_posteriors]
       [--logfile <logfile>]
       [--cpu]
       [--debug]
       [--vmp]

    * Use one of the following options to specify images to segment:
      1. --i <image_path> or 
      2. --dataset_list_file <dataset.yaml> --cohort <train|validation|test>
    * Options --i <image_path> and --dataset_list_file <dataset.yaml> are mutually exclusive.
"""

def main():
    args = argument_parse()
    
    # setup and configure root and main logger
    logfile = args.logfile if (args.logfile is not None) else os.path.join(os.getcwd(), "freeseg_predict.log")
    utils.config_logger(logfile=logfile, mode='w')
    mainlogger = logging.getLogger(__name__)
    mainlogger.addHandler(logging.StreamHandler())

    checkpoint = args.checkpoint    
    if (checkpoint is not None):
        if not os.path.isfile(checkpoint):
            mainlogger.error('ERROR: file does not exist: %s' % checkpoint)
            sys.exit(1)
                          
    # print the command
    cmd = ' '.join(sys.argv)
    cmdopts = cmd.split("--")
    mainlogger.info("")
    mainlogger.info("CWD: " + os.getcwd())
    mainlogger.info("CMD: " + "\n\t--".join(cmdopts))

    if (args.cpu):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ((args.i is not None) and (args.dataset_list_file is not None)):
        mainlogger.error("Options --i <image_path> and --dataset_list_file <dataset.yaml> are mutually exclusive")
        return

    assert ((args.i is not None) or (args.dataset_list_file is not None)), \
        "Use --i or --dataset_list_file to specify images to segment"
    if ((args.dataset_list_file is not None) and (args.cohort is None)):
        mainlogger.error("Use --cohort <train|validation|test> which dataset to segment")
        return

    codenames = None
    path_images = args.i
    path_gt = args.gt
    path_priors = args.prior
    if ((args.dataset_list_file is not None)):
        # --label <> needs to specified separately, pointing to a directory
        dataset_dict = Config.load_dataset_list(args.dataset_list_file)

        dataset = Config.retrieve_dataset_cohorts(dataset_dict, args.cohort)
        if (not dataset):
            mainlogger.info("Empty cohort, nothing to do.")
            return

        path_images, path_gt, path_priors, codenames = [], [], [], []
        for item in dataset:
            path_images.append(item["image_filepath"])
            path_gt.append(item["label_filepath"])
            if (item.get("prior_filepath")):
                path_priors.append(item["prior_filepath"])
            if (item.get("codename")):
                codenames.append(item["codename"])

        assert (len(path_images) == len(path_gt)), "image and label need to be the same length"                
        path_priors = path_priors if (len(path_priors)) else None
        codenames = codenames if (len(codenames)) else None
        if (path_priors is not None):
            assert (len(path_images) == len(path_priors)), "images and priors need to be the same length"
        if (codenames is not None):
            assert (len(path_images) == len(codenames)), "images and codenames need to be the same length"

    mainlogger.info("prediction in progress ...")
    if (logfile is not None):
        mainlogger.info(f"prediction log can be found in {logfile}")

    segmentation_names = None
    if (args.segmentation_names is not None):    
        segmentation_names=np.load(args.segmentation_names)
    predict(path_images, args.o, checkpoint,
            crop_size=args.crop_size,
            target_res=args.target_res,
            ctab=args.ctab,
            path_labels=args.label,
            path_priors=path_priors,
            codenames=codenames,
            path_gt=path_gt,
            addctab=True if (not args.noaddctab) else False,
            write_posteriors=args.write_posteriors,
            path_volumes=args.vol,
            keepgeom=(not args.nokeepgeom),
            device=device,
            debug=args.debug,
            segmentation_names=segmentation_names
            )

    # check memory usage
    if (args.vmp):
        utils.print_vm_peak()

    mainlogger.info("Done!")
    

def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=description)

    # input/outputs
    parser.add_argument("--i", type=str, help="Image(s) to segment. Can be a path to an image or to a folder.")
    parser.add_argument("--dataset_list_file", type=str, help="Image(s) to segment. Can be a path to an image or to a folder.")
    parser.add_argument("--cohort", nargs="+", type=str, default=['test'], help="Dataset cohort. Can be combinations of train, validation, or test")
    parser.add_argument("--o", type=str, required=True, help="Segmentation output(s). Must be a folder if --i designates a folder, or --dataset_list_file is specified.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint file to resume training from")
    parser.add_argument("--crop_size", nargs="+", type=int, help="Crop size for training and validation")
    parser.add_argument("--target_res", nargs="+", type=float, help="Segmentation output resolution")
    parser.add_argument("--nokeepgeom", action="store_true", help="Donot resample output to be the same as input geometry")
    parser.add_argument("--ctab", type=str, help="Path to the lookup table")
    parser.add_argument("--label", type=str, help="Label(s) for input image(s). Can be a path to a label or to a folder. The labels can be binary masks.")
    parser.add_argument("--prior", type=str, help="Input priors")
    parser.add_argument("--gt", type=str, help="Path to ground truth folder for dice evaluation.")
    parser.add_argument("--noaddctab", action="store_true", help="Do not embed colortable into seg output")
    parser.add_argument("--segmentation_names", type=str, help="Path to npy containing segmentation names corresponding to segmentation labels")
    parser.add_argument("--write_posteriors", action='store_true', help="Save the label posteriors.")
    parser.add_argument("--vol", type=str, help="Output for calculated label volumes.")
    parser.add_argument('--logfile', type=str, help='Set logfile (default is ./freeseg_predict.log)')
    parser.add_argument("--cpu", action='store_true', help="Run on CPU.")
    parser.add_argument("--debug", action='store_true', help="Output volumes for debugging.")
    parser.add_argument('--vmp', action='store_true', help='Enable printing of vmpeak at the end.')

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    # parse commandline
    args = parser.parse_args()

    return args


def predict(path_images, out_segmentations, checkpoint, crop_size=None, target_res=None, ctab=None, path_labels=None, path_priors=None, codenames=None,
            path_gt=None, addctab=True, write_posteriors=False, path_volumes=None, device=None, debug=False, keepgeom=False, segmentation_names=None):
    prediction = Prediction(device, ctab=ctab, debug=debug)
    prediction.load_model(checkpoint)
    prediction.predict(path_images, out_segmentations,
                       crop_size=crop_size,
                       target_res=target_res,
                       path_labels=path_labels,
                       path_priors=path_priors,
                       codenames=codenames,
                       path_gt=path_gt,
                       addctab=addctab,
                       write_posteriors=write_posteriors,
                       path_volumes=path_volumes,
                       keepgeom=keepgeom,
                       segmentation_names=segmentation_names)


# execute script
if __name__ == '__main__':
    main()
