#!/bin/bash
#SBATCH -J s3m
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/research/S3M/out/nb_10frames_maxmeansim_a0.6_%j.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracesim

python /home/mdafifal.mamun/research/S3M/src/pretrain.py
python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json --bucket eclipse --trim_len 2 --lang java --multi_stack
python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json --bucket eclipse --trim_len 0 --lang java --multi_stack

# python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans --trim_len 0 --lang java --multi_stack