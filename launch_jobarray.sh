#!/bin/bash

# Define the array of run names
run_names=(
    "dummy"
    "dummy_multiclass"
    "pitgland_binarized"
    "pitgland_cropped"
)

# Loop through the run names and submit jobs
for i in "${!run_names[@]}"; do
    # Calculate the array task ID (adding 1 because array jobs start at 1)
    task_id=$((i + 1))
    
    # Submit the job with the specific output file name
    sbatch --account=fsm \
           --output="new_run/slurm_output/train_%A_%a_${run_names[i]}.log" \
           --error="new_run/slurm_output/train_%A_%a_${run_names[i]}.log" \
           --array=$task_id \
           jobarray_train.sh
done