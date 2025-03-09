#!/bin/bash
#SBATCH -J s3m
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/research/S3M/out/train_out/eclipse_bge_3500days_%j.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracesim


echo "all-mpnet-base-v2"
python /home/mdafifal.mamun/research/S3M/src/pretrain.py --plm all-mpnet-base-v2
python /home/mdafifal.mamun/research/S3M/src/main.py transformer \
    --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json \
    --bucket eclipse --trim_len 2 --lang java --multi_stack --loss ranknet \
    --encoder_path /work/disa_lab/afif/projects/deduplication/trained_embedding_models/all-mpnet-base-v2_eclipse_10frames_4train_2trim_3850_traindays/final


echo "DistilRoBERTa"
python /home/mdafifal.mamun/research/S3M/src/pretrain.py --plm all-distilroberta-v1
python /home/mdafifal.mamun/research/S3M/src/main.py transformer \
    --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json \
    --bucket eclipse --trim_len 2 --lang java --multi_stack --loss ranknet \
    --encoder_path /work/disa_lab/afif/projects/deduplication/trained_embedding_models/all-distilroberta-v1_eclipse_10frames_4train_2trim_3850_traindays/final


echo "miniLM"
python /home/mdafifal.mamun/research/S3M/src/pretrain.py --plm all-MiniLM-L12-v2
python /home/mdafifal.mamun/research/S3M/src/main.py transformer \
    --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json \
    --bucket eclipse --trim_len 2 --lang java --multi_stack --loss ranknet \
    --encoder_path /work/disa_lab/afif/projects/deduplication/trained_embedding_models/all-minilm-l12-v2_eclipse_10frames_4train_2trim_3850_traindays/final