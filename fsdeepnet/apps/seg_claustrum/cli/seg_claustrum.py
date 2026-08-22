#!/usr/bin/env python

import os
import sys
import argparse


def main():

    # - replace '-i' with '--i'
    # - replace '-o' with '--o'
    # - replace '-m', '--model' with '--checkpoint'
    # - replace '-c', '--csv' with '--vol'
    # - replace '-p', '--post' with '--write_posteriors'
    #   remove argument following '-p', --post'
    # - add the following arguments:
    #     --nokeepgeom --keep_biggest_component --use_topology_classes --smooth_posteriors --resamplefirst --logfile <>
    for arg in ["-i", "-o", "-m", "--model", "-c", "--csv", "-p", "--post"]:
        if (arg in sys.argv):
            index = sys.argv.index(arg)
            if (arg == "-i"):
                sys.argv[index] = "--i"
            elif (arg == "-o"):
                sys.argv[index] = "--o"
            elif (arg == "-m" or arg == "--model"):
                sys.argv[index] = "--checkpoint"
            elif (arg == "-c" or arg == "--csv"):
                sys.argv[index] = "--vol"
            elif (arg == "-p" or arg == "--post"):
                sys.argv[index] = "--write_posteriors"
                sys.argv.pop(index+1)

    # pass default model name only
    sys.argv.extend(["--checkpoint", "claustrum_seg_20250616.pth"])

    outdir = None
    if ("--o" in sys.argv):
        index = sys.argv.index("--o")
        outdir = sys.argv[index+1]
        if (os.path.isfile(outdir)):
            outdir = os.path.dirname(outdir)
    extra_args = ["--nokeepgeom",  
                  "--keep_biggest_component", 
                  "--use_topology_classes", 
                  "--smooth_posteriors", 
                  "--resamplefirst",
                ]
    if (outdir is not None):
        fsdeepnetlog = os.path.join(outdir, "log", "fsdeepnet_seg_claustrum.log")
        extra_args.extend(["--logfile", fsdeepnetlog])
    sys.argv.extend(extra_args)

    from fsdeepnet.cli.fsdeepnet_predict import main as fsdeepnet_predict_main
    sys.exit(fsdeepnet_predict_main())


# execute script
if __name__ == '__main__':
    main()


