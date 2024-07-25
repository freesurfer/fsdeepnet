#!/bin/bash

#SBATCH --account=fsm
#SBATCH --partition=rtx6000,rtx8000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=new_run/slurm_output/train_%A_%a.log  # Standard output log
#SBATCH --error=new_run/slurm_output/train_%A_%a.log   # Standard error log
#SBATCH --job-name=pitgland_training_%a
#SBATCH --array=1-4  # Number of jobs in the array

set -x

# print current working directory
# echo $PWD

# Create the slurm_output directory if it doesn't exist
mkdir -p slurm_output

# Define arrays for output folders and config files
output_folders=(
    "new_run/output/train_dummy"
    # "new_run/output/train_dummy_multiclass_binarized"
    "new_run/output/train_dummy_multiclass"
    # "new_run/output/train_pitgland_300_binarized"
    # "new_run/output/train_pitgland_300_multiclass"
    "new_run/output/train_pitgland_binarized"
    "new_run/output/train_pitgland_cropped"
)

dataset_list_files=(
    "configs/train_dummy_dataset_list.yaml"
    # "configs/train_dummy_multiclass_binarized_dataset_list.yaml"
    "configs/train_dummy_multiclass_dataset_list.yaml"
    # "configs/train_pitgland_300_binarized_dataset_list.yaml"
    # "configs/train_pitgland_300_multiclass_dataset_list.yaml"
    "configs/train_pitgland_binarized_dataset_list.yaml"
    "configs/train_pitgland_cropped_dataset_list.yaml"
)

config_files=(
    "configs/config_dummy.yaml"
    # "configs/config_dummy_multiclass_binarized.yaml"
    "configs/config_dummy_multiclass.yaml"
    # "configs/config_pitgland_300_binarized.yaml"
    # "configs/config_pitgland_300_multiclass.yaml"
    "configs/config_pitgland_binarized.yaml"
    "configs/config_pitgland_cropped.yaml"
    )

run_names=(
    "dummy"
    # "dummy_multiclass_binarized"
    "dummy_multiclass"
    # "pitgland_300_binarized"
    # "pitgland_300_multiclass"
    "pitgland_binarized"
    "pitgland_cropped"
)

# Get the index of the current job in the array
index=$((SLURM_ARRAY_TASK_ID - 1))

# Get the corresponding output folder and config file for the current job
output_folder="${output_folders[$index]}"
dataset_list_file="${dataset_list_files[$index]}"
config_file="${config_files[$index]}"

/autofs/space/curv_001/users/avnish/miniconda3/envs/pgland/bin/python train.py \
--output_folder "$output_folder" \
--config "$config_file" \
--dataset_list_file "$dataset_list_file" \
--run_name "${run_names[$index]}"