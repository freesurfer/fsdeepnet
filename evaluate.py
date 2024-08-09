import sys
import json
import argparse

from evaluation import Evaluation

"""
Usage: evaluate.py 
       --label_mapping <label_mapping.json>
       --gt <ground_truth>
       --seg <segmentation>
       [--path_dice <path_dice>]

python -m pdb evaluate.py --label_mapping /space/metropolis/1/users/yh887/trainings/freeseg/pitgland_cropped/train/restructure//restructure.metropolis.model_arch_dict+label_lookup/label_mapping.json --gt /space/metropolis/1/users/yh887/trainings/freeseg/data/pitgland_cropped/labels/ --seg /space/metropolis/1/users/yh887/freeseg.restructure/test.predict+evaluate/ --path_dice /space/metropolis/1/users/yh887/freeseg.restructure/test.predict+evaluate/dices.freeseg-evaluate.npy
"""

def main():
    args = argument_parse()

    with open(args.label_mapping, "r") as f:
        label_mapping = json.load(f)

    # Ensure keys in label_mapping are integers
    label_mapping = {int(k): v for k, v in label_mapping.items()}
    labels_segmentation = [label for label, idx in label_mapping.items()]
    
    evaluate(labels_segmentation, args.gt, args.seg, args.path_dice)


# ??? todo: replace label_mapping with something else ???
def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--label_mapping", type=str, required=True, help="Path to the label_mapping.json file. If not provided, the script will search for it in the model checkpoint directory.")    
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth (folder) for dice evaluation.")
    parser.add_argument("--seg", type=str, required=True, help="Path to segmentation (folder) for dice evaluation.")
    parser.add_argument("--path_dice", type=str, help="Path to dice scores output.")

    # parse commandline
    args = parser.parse_args()

    return args


def evaluate(labels_segmentation, gt, seg, path_dice=None):
    eval = Evaluation(labels_segmentation)
    eval.evaluate(gt, seg, path_dice=path_dice)

    
# execute script
if __name__ == '__main__':
    main()
    
