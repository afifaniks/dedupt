#!/bin/bash
#SBATCH -J s3m
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=200G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/research/S3M/out/pretrain_out/gnome_multi.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracesim

# echo "Pretraining"
python /home/mdafifal.mamun/research/S3M/src/pretrain.py