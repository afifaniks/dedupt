#!/bin/bash
#SBATCH -J s3m
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/research/S3M/out/train_out/deepcrash_ec_nb_lstm1_%j.out

CONFIG_PARAM=$1
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracesim

echo "Running deepcrash on eclipse..."
python /home/mdafifal.mamun/research/S3M/src/main.py deepcrash --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json --bucket_name eclipse --trim_len 2 --train_days 3850 --test_days 700 --val_days 140 --warmup_days 350

echo "Running deepcrash on netbeans..."
python /home/mdafifal.mamun/research/S3M/src/main.py deepcrash --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans --train_days 3850 --test_days 700 --val_days 140 --warmup_days 350

# echo "Running deepcrash on gnome..."
# python /home/mdafifal.mamun/research/S3M/src/main.py deepcrash --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/gnome_2011/gnome_stacktraces_filtered_98_1_2.json --bucket_name gnome --lang cpp --train_days 3850 --test_days 700 --val_days 140 --warmup_days 350 --skip_training

# echo "Running deepcrash on ubuntu..."
# python /home/mdafifal.mamun/research/S3M/src/main.py deepcrash --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/campbell_dataset/campbell_stacktraces.json --bucket_name ubuntu --lang cpp --train_days 2000 --test_days 700 --val_days 50 --warmup_days 350