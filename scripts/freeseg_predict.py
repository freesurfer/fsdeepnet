#!/usr/bin/env python

import os
import torch
import argparse
import yaml

from freeseg.prediction import Prediction


"""
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
       [--cpu]
       [--debug]

    * Use one of the following options to specify images to segment:
      1. --i <image_path> or 
      2. --dataset_list_file <dataset.yaml> --cohort <train|validation|test>
    * Options --i <image_path> and --dataset_list_file <dataset.yaml> are mutually exclusive.
"""

def main():
    args = argument_parse()
    
    if (args.cpu):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ((args.i is not None) and (args.dataset_list_file is not None)):
        print("Options --i <image_path> and --dataset_list_file <dataset.yaml> are mutually exclusive")
        return

    assert ((args.i is not None) or (args.dataset_list_file is not None)), \
        "Use --i or --dataset_list_file to specify images to segment"
    if ((args.dataset_list_file is not None) and (args.cohort is None)):
        print("Use --cohort <train|validation|test> which dataset to segment")
        return

    path_images = args.i
    path_gt = args.gt
    path_priors = args.prior
    if ((args.dataset_list_file is not None)):
        # --label <> needs to specified separately, pointing to a directory
        with open(args.dataset_list_file, "r") as file:
            dataset_dict = yaml.safe_load(file)

        dataset = dataset_dict.get(args.cohort)
        path_images, path_gt, path_priors = [], [], []
        for item in dataset:
            path_images.append(item["image_filepath"])
            path_gt.append(item["label_filepath"])
            if (item.get("prior_filepath")):
                path_priors.append(item["prior_filepath"])

        path_priors = path_priors if (len(path_priors)) else None
        assert (len(path_images) == len(path_gt)), "image and label need to be the same length"
        if (path_priors is not None):
            assert (len(path_images) == len(path_priors)), "images and priors need to be the same length"
        
    predict(path_images, args.o, args.checkpoint,
            crop_size=args.crop_size,
            ctab=args.ctab,
            path_labels=args.label,
            path_priors=path_priors,
            path_gt=path_gt,
            addctab=True if (not args.noaddctab) else False,
            write_posteriors=args.write_posteriors,
            device=device,
            debug=args.debug)
    

def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--i", type=str, help="Image(s) to segment. Can be a path to an image or to a folder.")
    parser.add_argument("--dataset_list_file", type=str, help="Image(s) to segment. Can be a path to an image or to a folder.")
    parser.add_argument("--cohort", type=str, help="Dataset cohort. Can be train, validation, or test")
    parser.add_argument("--o", type=str, required=True, help="Segmentation output(s). Must be a folder if --i designates a folder, or --dataset_list_file is specified.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint file to resume training from")
    parser.add_argument("--crop_size", nargs="+", type=int, help="Crop size for training and validation")
    parser.add_argument("--ctab", type=str, help="Path to the lookup table")
    parser.add_argument("--label", type=str, help="Label(s) for input image(s). Can be a path to a label or to a folder. The labels can be binary masks.")
    parser.add_argument("--prior", type=str, help="Input priors")
    parser.add_argument("--gt", type=str, help="Path to ground truth folder for dice evaluation.")
    parser.add_argument("--noaddctab", action="store_true", help="Do not embed colortable into seg output")
    parser.add_argument("--write_posteriors", action='store_true', help="Save the label posteriors.")
    parser.add_argument("--cpu", action='store_true', help="Run on CPU.")
    parser.add_argument("--debug", action='store_true', help="Output volumes for debugging.")

    # parse commandline
    args = parser.parse_args()

    return args


def predict(path_images, out_segmentations, checkpoint, crop_size=None, ctab=None, path_labels=None, path_priors=None,
            path_gt=None, addctab=True, write_posteriors=False, device=None, debug=False):
    prediction = Prediction(device, ctab=ctab)
    prediction.load_model(checkpoint)
    prediction.predict(path_images, out_segmentations,
                       crop_size=crop_size,
                       path_labels=path_labels,
                       path_priors=path_priors,
                       path_gt=path_gt,
                       addctab=addctab,
                       write_posteriors=write_posteriors,
                       debug=debug)


# execute script
if __name__ == '__main__':
    main()
