#!/bin/bash
#SBATCH -J s3m
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/research/S3M/out/train_out/s3m_gnome_3850days_%j.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracesim


echo "Training"

python /home/mdafifal.mamun/research/S3M/src/main.py s3m --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/gnome_2011/gnome_stacktraces_filtered_98_1_2.json --bucket_name gnome --lang cpp --loss ranknet
