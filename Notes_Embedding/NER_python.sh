#!/bin/bash

#SBATCH --job-name=group_emb_all_patients
#SBATCH --output=b_output_%j.txt # Standard output
#SBATCH --error=b_error_%j.txt   # Standard error

#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=32768

python group_emb.py