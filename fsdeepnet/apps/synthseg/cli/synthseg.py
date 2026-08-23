#!/usr/bin/env python

import os
import sys
import argparse


def main():

    # if '--keepgeom' is not found, add '--nokeepgeom'
    if ("--keepgeom" not in sys.argv):
        sys.argv.append("--nokeepgeom")
    # if '--addctab' is not found, add '--noaddctab'
    if ("--addctab" not in sys.argv):
        sys.argv.append("--noaddctab")

    # - replace '--model' with '--checkpoint'
    # - replace '--crop' with '--crop_size'
    # - replace '--post' with '--write_posteriors'
    #   remove argument following '-p', --post'
    # - remove '--keepgeom'
    # - remove '--addctab'
    for arg in ["--model", "--crop", "--post", "--keepgeom", "--addctab"]:
        if (arg in sys.argv):
            index = sys.argv.index(arg)
            if (arg == "--model"):
                sys.argv[index] = "--checkpoint"
            elif (arg == "--crop"):
                sys.argv[index] = "--crop_size"
            elif (arg == "--post"):
                sys.argv[index] = "--write_posteriors"
                sys.argv.pop(index+1)
            elif (arg == "--keepgeom" or arg == "--addctab"):
                sys.argv.pop(index)

    # pass default model, name only
    if ("--checkpoint" not in sys.argv):
        sys.argv.append("--checkpoint")
        sys.argv.append("synthseg_2.0.pth")

    # - add the following arguments:
    #     --keep_biggest_component, --smooth_posteriors, --use_topology_classes, --flip
    sys.argv.append("--keep_biggest_component")
    sys.argv.append("--smooth_posteriors")
    if ("--fast" not in sys.argv):
        sys.argv.append("--use_topology_classes")
        sys.argv.append("--flip")

    from fsdeepnet.cli.fsdeepnet_predict import main as fsdeepnet_predict_main
    retcode = fsdeepnet_predict_main()
    if (retcode == 0):
        print('\nIf you use this tool in a publication, please cite:')
        print('SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining')
        print('B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias')
        print('Medical Image Analysis, 2023.')
        if ("--parc" in sys.argv):
            print('Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets')
            print('B. Billot, M. Colin, Y. Cheng, S.E. Arnold, S. Das, J.E. Iglesias')
            print('PNAS, 2023.')
    sys.exit(retcode)


# execute script
if __name__ == '__main__':
    main()


