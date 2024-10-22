#!/bin/bash

#SBATCH --account=fsm
#SBATCH --partition=rtx6000,rtx8000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_output/train_basic_%j.log  # Standard output log
#SBATCH --error=slurm_output/train_basic_%j.log   # Standard error log
#SBATCH --job-name=pitgland_training

set -x

# Activate conda environment
# source activate pgland

# print current working directory
echo $PWD

# Create the slurm_output directory if it doesn't exist
mkdir -p slurm_output

/autofs/space/curv_001/users/avnish/miniconda3/envs/pgland/bin/python train.py --output_folder output/train_pitgland_300_binarized_DEBUG
