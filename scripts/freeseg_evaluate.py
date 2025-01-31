#!/usr/bin/env python

import sys
import logging
import numpy as np
import argparse

from freeseg.utils import config_logger
from freeseg.evaluation import Evaluation

"""
Usage: freeseg_evaluate.py 
       --gt <ground_truth>
       --seg <segmentation>
       [--segmentation_labels <segmentation_labels.npy>]
       [--evaluation_labels <label1 label2 ...>]
       [--path_dice <path_dice>]
       [--logfile <logfile>]
"""

def main():
    args = argument_parse()

    # setup and configure logging
    config_logger(args.logfile, logging.DEBUG, "%(asctime)s [%(levelname)s] %(message)s")
    # print the command
    logging.info(' '.join(sys.argv))

    labels_segmentation = None
    if (args.segmentation_labels is not None):
        labels_segmentation = np.load(args.segmentation_labels)
    if (args.evaluation_labels is not None):
        labels_segmentation = args.evaluation_labels

    assert labels_segmentation is not None, 'please specify labels for dice evaluation using either --segmentation_labels <segmentation_labels.npy> or --evaluation_labels <label1 label2 ...>'
        

    # evaluate() expects labels_segmentation as a list [] or 1D numpy array
    evaluate(labels_segmentation, args.gt, args.seg, args.path_dice)


def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--segmentation_labels", type=str, help="Path to segmentation_labels.npy")
    parser.add_argument("--evaluation_labels", nargs="+", type=int, help="Labels for dice evaluation")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth (folder) for dice evaluation.")
    parser.add_argument("--seg", type=str, required=True, help="Path to segmentation (folder) for dice evaluation.")
    parser.add_argument("--path_dice", type=str, help="Path to dice scores output.")
    parser.add_argument('--logfile', type=str, default='freeseg_evaluate.log', help='Set logfile (default is freeseg_evaluate.log)')

    # parse commandline
    args = parser.parse_args()

    return args


def evaluate(labels_segmentation, gt, seg, path_dice=None):
    eval = Evaluation(labels_segmentation)
    eval.evaluate(gt, seg, path_dice=path_dice)

    
# execute script
if __name__ == '__main__':
    main()
    
