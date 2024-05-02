#!/bin/bash

#SBATCH --partition=gpu-a100   # Specify the exact GPU partition
#SBATCH --gres=gpu      # Requesting one NVIDIA A100 GPU
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # Number of tasks (keep at 1 if not using MPI)
#SBATCH --cpus-per-task=32     # Number of cores per task
#SBATCH --mem=256G             # Memory per node
#SBATCH --time=24:00:00        # Time limit hrs:min:sec
#SBATCH --output=%x_output_%j.txt  # Standard output and error log
#SBATCH --error=%x_error_%j.txt




#SBATCH --job-name=SAGE_MIMIC_8_1_2_150   # Name of the job

python PLP_SAGE.py
# python PLP_GCN.py
