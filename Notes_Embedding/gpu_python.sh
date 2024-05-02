#!/bin/bash

#SBATCH --output=b_output_%j.txt # Standard output
#SBATCH --error=b_error_%j.txt   # Standard error

#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=gpu --gres gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32768

#SBATCH --job-name=clin_ner_gpu

python spacy_clin_ner_gpu.py