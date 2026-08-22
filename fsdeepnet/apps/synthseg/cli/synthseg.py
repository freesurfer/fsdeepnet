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

    # - add the following arguments:
    #     --keep_biggest_component, --smooth_posteriors, --use_topology_classes, --flip
    sys.argv.append("--keep_biggest_component")
    sys.argv.append("--smooth_posteriors")
    if ("--fast" not in sys.argv):
        sys.argv.append("--use_topology_classes")
        sys.argv.append("--flip")

    from fsdeepnet.cli.fsdeepnet_predict import main as fsdeepnet_predict_main
    sys.exit(fsdeepnet_predict_main())


# execute script
if __name__ == '__main__':
    main()


