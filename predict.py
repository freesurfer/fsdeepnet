import os
import torch
import argparse

from prediction import Prediction


"""
Usage: predict.py 
       --i <input_images>
       --o  <output_segmentations>
       --checkpoint <checkpoint>
       [--crop_size <W H D>]
       [--gt <ground_truth_dir>] 
       [--path_dice <path_dice>]
       [--noaddctab]
       [--write_posteriors]
       [--cpu]
"""

def main():
    args = argument_parse()
    
    if (args.cpu):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict(args.i, args.o, args.checkpoint, args.crop_size,
            path_gt=args.gt,
            path_dice=args.path_dice,
            addctab=True if (not args.noaddctab) else False,
            write_posteriors=args.write_posteriors,
            device=device)
    

def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--i", type=str, required=True, help="Image(s) to segment. Can be a path to an image or to a folder.")
    parser.add_argument("--o", type=str, required=True, help="Segmentation output(s). Must be a folder if --i designates a folder.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint file to resume training from")
    parser.add_argument("--crop_size", nargs="+", type=int, help="Crop size for training and validation")
    parser.add_argument("--gt", type=str, help="Path to ground truth folder for dice evaluation.")
    parser.add_argument("--path_dice", type=str, help="Path to dice scores output.")
    parser.add_argument("--noaddctab", action="store_true", help="Do not embed colortable into seg output")
    parser.add_argument("--write_posteriors", action='store_true', help="Save the label posteriors.")
    parser.add_argument("--cpu", action='store_true', help="Run on CPU.")    

    # parse commandline
    args = parser.parse_args()

    return args


def predict(path_images, out_segmentations, checkpoint, crop_size,
            path_gt=None, path_dice=None, addctab=True, write_posteriors=None, device=None):
    prediction = Prediction(device)
    prediction.load_model(checkpoint)
    prediction.predict(path_images, out_segmentations, crop_size,
                       path_gt=path_gt,
                       path_dice=path_dice,
                       addctab=addctab,
                       write_posteriors=write_posteriors)


# execute script
if __name__ == '__main__':
    main()
