#!/bin/bash
#SBATCH -J s3m
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/research/S3M/out/train_out/deepcrash_eclipse_netbeans_3850days_%j.out

CONFIG_PARAM=$1
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracesim

echo "Running deepcrash on netbeans..."
python /home/mdafifal.mamun/research/S3M/src/main.py deepcrash --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans

echo "Running deepcrash on eclipse..."
python /home/mdafifal.mamun/research/S3M/src/main.py deepcrash --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json --bucket_name eclipse