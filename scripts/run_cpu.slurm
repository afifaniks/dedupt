#!/bin/bash
#SBATCH -J s3m
#SBATCH -t 4-00:00:00
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH --partition=disa
#SBATCH --output=/home/mdafifal.mamun/research/S3M/out/eclipse_classic_%j.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracesim

python /home/mdafifal.mamun/research/S3M/src/main.py all-classic --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json