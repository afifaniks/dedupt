#!/bin/bash
#SBATCH -J s3m
#SBATCH -t 2-00:00:00
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --partition=cpu2023
#SBATCH --output=/home/mdafifal.mamun/research/S3M/out/eclipse_durfex_%j.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracesim

python /home/mdafifal.mamun/research/S3M/src/main.py durfex --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json --bucket_name eclipse
python /home/mdafifal.mamun/research/S3M/src/main.py durfex --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans