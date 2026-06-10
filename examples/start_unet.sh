#!/bin/bash -l

#SBATCH -A [NAME OF PROJECT]
#SBATCH -t 40:00:00
#SBATCH -J train_unet
#SBATCH -C gpu
#SBATCH --gres=gpu:1

module load conda
export CONDA_ENVS_PATH=[path/to/project]
source conda_init.sh
conda activate dl-env
python train_new_unet.py

