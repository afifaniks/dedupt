#!/bin/bash
#SBATCH -J s3m
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/research/S3M/out/train_out/s3m_ubuntu_auc_2000days_%j.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracesim


echo "Training"

# python /home/mdafifal.mamun/research/S3M/src/pretrain.py
# python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/campbell_dataset/campbell_stacktraces.json --bucket_name ubuntu --lang cpp --loss ranknet --encoder_path /work/disa_lab/afif/projects/deduplication/trained_embedding_models/baai_bge-base-en-v1.5_ubuntu_50frames_5train_0trim_2000_traindays/final
python /home/mdafifal.mamun/research/S3M/src/main.py s3m --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/campbell_dataset/campbell_stacktraces.json --bucket_name ubuntu --lang cpp --loss ranknet --encoder_path none_s