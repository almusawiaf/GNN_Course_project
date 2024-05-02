#!/bin/bash

#SBATCH --job-name=note_embedding
#SBATCH --output=b_output_%j.txt # Standard output
#SBATCH --error=b_error_%j.txt   # Standard error

#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=gpu --gres gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32768

# module load miniconda3/py39_23.9.0
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate myCUDA_transformer
# module load cuda/12.3

jupyter nbconvert --to notebook --execute GPU_note_embeddings.ipynb --output b_output_%j.ipynb

# conda deactivate
