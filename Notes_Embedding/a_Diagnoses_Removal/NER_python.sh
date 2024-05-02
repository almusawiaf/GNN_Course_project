#!/bin/bash

#SBATCH --output=b_output_%j.txt # Standard output
#SBATCH --error=b_error_%j.txt   # Standard error

#SBATCH --partition=cpu-small    
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1  
#SBATCH --mem=80G  


#SBATCH --job-name=concurrent_clin_ner_1_all

# module load Anaconda3/2023.09-0          # Adjust based on available modules for Python environment
source activate env_sci_spacy            

python concurrent_clin_ner_1.py
