import os
import json
import torch
import logging
import argparse

from prediction import Prediction


"""
Usage: predict.py 
       --i <input_images>
       --o  <output_segmentations>
       --checkpoint <checkpoint>
       --crop_size <W H D>
       [--gt <ground_truth_dir>] 
       [--path_dice <path_dice>]
       [--label_mapping <label_mapping.json>]
       [--addctab]
       [--write_posteriors]
       [--cpu]

       * config.yaml need to have the same network parameters as training.
         If it is not given, config.yaml saved in the training root directory is used.
       * If <label_mapping.json> is not given, label_mapping.json in the training root directory is used.

python -m pdb predict.py --i /space/metropolis/1/users/yh887/trainings/freeseg/data/pitgland_cropped/images --o /space/metropolis/1/users/yh887/freeseg.restructure/test.predict+evaluate2 --checkpoint /space/metropolis/1/users/yh887/trainings/freeseg/pitgland_cropped/train/restructure//restructure.metropolis.model_arch_dict+label_lookup/checkpoints/dice_100_train_loss0.2551_train_dice0.7455.pth  --crop_size 64 64 64 --write_posteriors --gt /space/metropolis/1/users/yh887/trainings/freeseg/data/pitgland_cropped/labels/

python -m pdb predict.py --i /space/metropolis/1/users/yh887/trainings/freeseg/data/pitgland_cropped/images --o /space/metropolis/1/users/yh887/freeseg.restructure/test.predict+evaluate3 --checkpoint /space/metropolis/1/users/yh887/trainings/freeseg/pitgland_cropped/train/restructure//restructure.metropolis.model_arch_dict+label_lookup/checkpoints/dice_100_train_loss0.2551_train_dice0.7455.pth  --crop_size 64 64 64 --write_posteriors --gt /space/metropolis/1/users/yh887/trainings/freeseg/data/pitgland_cropped/labels/ --path_dice /space/metropolis/1/users/yh887/freeseg.restructure/test.predict+evaluate3/dices.freeseg-predict+evaluate.npy
"""

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,  # Set the log level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to terminal
    ],
)

def main():
    args = argument_parse()
    
    if (args.cpu):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ??? todo: remove dependency on label_mapping.json ???
    if args.label_mapping:
        label_mapping_path = args.label_mapping
    else:
        # Search for label_mapping.json in the root dir of training output
        model_checkpoint_dir = os.path.dirname(args.checkpoint)
        label_mapping_path = os.path.join(model_checkpoint_dir, "..", "label_mapping.json")

    with open(label_mapping_path, "r") as f:
        label_mapping = json.load(f)

    # Ensure keys in label_mapping are integers
    label_mapping = {int(k): v for k, v in label_mapping.items()}

    predict(args.i, args.o, label_mapping, args.checkpoint, args.crop_size,
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
    parser.add_argument("--crop_size", nargs="+", type=int, required=True, help="Crop size for training and validation")
    parser.add_argument("--gt", type=str, help="Path to ground truth folder for dice evaluation.")
    parser.add_argument("--path_dice", type=str, help="Path to dice scores output.")
    parser.add_argument("--noaddctab", action="store_true", help="Do not embed colortable into seg output")
    parser.add_argument("--write_posteriors", action='store_true', help="Save the label posteriors.")
    parser.add_argument("--cpu", action='store_true', help="Run on CPU.")    
    parser.add_argument("--label_mapping", type=str, default=None, help="Path to the label_mapping.json file. If not provided, the script will search for it in the model checkpoint directory.")    

    # parse commandline
    args = parser.parse_args()

    return args


# ??? todo: remove label_mapping
def predict(path_images, out_segmentations, label_mapping, checkpoint, crop_size,
            path_gt=None, path_dice=None, addctab=True, write_posteriors=None, device=None):
    prediction = Prediction(label_mapping, device)
    prediction.load_model(checkpoint)
    prediction.predict(path_images, out_segmentations, crop_size,
                       path_gt=path_gt,
                       path_dice=path_dice,
                       addctab=addctab,
                       write_posteriors=write_posteriors)


# execute script
if __name__ == '__main__':
    main()
