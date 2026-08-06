#!/usr/bin/env python

import os
import sys
import logging
import numpy as np
import argparse

from fsdeepnet.utils import utility as utils
from fsdeepnet.evaluation import Evaluation

"""
Usage: fsdeepnet_evaluate.py 
       --gt <ground_truth>
       --seg <segmentation>
       [--segmentation_labels <segmentation_labels.npy>]
       [--evaluation_labels <label1 label2 ...>]
       [--path_dice <path_dice>]
       [--logfile <logfile>]
"""

def main():
    args = argument_parse()

    # setup and configure root and main logger
    logfile = args.logfile if (args.logfile is not None) else os.path.join(os.getcwd(), "fsdeepnet_evaluate.log")
    utils.config_logger(logfile=logfile, mode='w')
    mainlogger = logging.getLogger(__name__)
    mainlogger.addHandler(logging.StreamHandler())

    # print the command
    mainlogger.info("")
    mainlogger.info("CWD: " + os.getcwd())
    mainlogger.info(' '.join(sys.argv))

    labels_segmentation = None
    if (args.segmentation_labels is not None):
        labels_segmentation = np.load(args.segmentation_labels)
    if (args.evaluation_labels is not None):
        labels_segmentation = args.evaluation_labels

    assert labels_segmentation is not None, 'please specify labels for dice evaluation using either --segmentation_labels <segmentation_labels.npy> or --evaluation_labels <label1 label2 ...>'
        
    mainlogger.info("evaluation in progress ...")
    if (logfile is not None):
        mainlogger.info(f"evaluation log can be found in {logfile}")

    # evaluate() expects labels_segmentation as a list [] or 1D numpy array
    evaluate(labels_segmentation, args.gt, args.seg, args.path_dice)
    mainlogger.info("Done!")


def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--segmentation_labels", type=str, help="Path to segmentation_labels.npy")
    parser.add_argument("--evaluation_labels", nargs="+", type=int, help="Labels for dice evaluation")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth (folder) for dice evaluation.")
    parser.add_argument("--seg", type=str, required=True, help="Path to segmentation (folder) for dice evaluation.")
    parser.add_argument("--path_dice", type=str, help="Path to dice scores output.")
    parser.add_argument('--logfile', type=str, help='Set logfile (default is ./fsdeepnet_evaluate.log)')

    # parse commandline
    args = parser.parse_args()

    return args


def evaluate(labels_segmentation, gt, seg, path_dice=None):
    eval = Evaluation(labels_segmentation)
    eval.evaluate(gt, seg, path_dice=path_dice)

    
# execute script
if __name__ == '__main__':
    main()
    
