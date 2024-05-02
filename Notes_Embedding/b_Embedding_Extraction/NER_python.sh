#!/bin/bash

#SBATCH --output=b_output_%j.txt # Standard output
#SBATCH --error=b_error_%j.txt   # Standard error

#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=gpu --gres gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32768

#SBATCH --job-name=emb_MODIFIED_2083180

# conda activate envCUDA

python GPU_note_emb_2.py