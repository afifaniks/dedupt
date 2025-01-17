#!/bin/bash
#SBATCH -J s3m
#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH --partition=gpu-a100 --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/research/S3M/out/nb_10frames_maxsim_%j.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracesim

# python /home/mdafifal.mamun/research/S3M/src/main.py s3m --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json
# python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json
# python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json --bucket eclipse --trim_len 1
# python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json --bucket eclipse --trim_len 0
# python /home/mdafifal.mamun/research/S3M/src/pretrain.py
# python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/gnome_2011/gnome_stacktraces_filtered.json --bucket_name gnome
# python /home/mdafifal.mamun/research/S3M/src/main.py s3m --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json --trim_len 1

# python /home/mdafifal.mamun/research/S3M/src/main.py s3m --data_path /home/mdafifal.mamun/research/S3M/src/netbeans_stacktraces.json --trim_len 1
# python /home/mdafifal.mamun/research/S3M/src/main.py s3m --data_path /home/mdafifal.mamun/research/S3M/src/netbeans_stacktraces.json --trim_len 1
# python /home/mdafifal.mamun/research/S3M/src/main.py s3m --data_path /home/mdafifal.mamun/research/S3M/src/netbeans_stacktraces.json --trim_len 2

# python /home/mdafifal.mamun/research/S3M/src/main.py all --data_path /home/mdafifal.mamun/research/S3M/src/netbeans_stacktraces.json

# python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json --bucket_name eclipse --trim_len 2 --lang java
python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans --trim_len 0 --lang java --multi_stack True