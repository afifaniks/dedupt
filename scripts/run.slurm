#!/bin/bash
#SBATCH -J s3m
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/research/S3M/out/train_out/20_frames_eclipse_netbeans_3850days_%j.out

CONFIG_PARAM=$1
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracesim

# # echo "Pretraining"
# python /home/mdafifal.mamun/research/S3M/src/main.py transformer \
#     --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/netbeans_2016/netbeans_stacktraces.json \
#     --bucket_name netbeans --trim_len 0 --lang java --multi_stack --loss ranknet \
#     --encoder_path /work/disa_lab/afif/projects/deduplication/trained_embedding_models/baai_bge-base-en_netbeans_10frames_5train_0trim_3850_traindays/final

# python /home/mdafifal.mamun/research/S3M/src/main.py transformer \
#     --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/netbeans_2016/netbeans_stacktraces.json \
#     --bucket_name netbeans --trim_len 0 --lang java --multi_stack --loss ranknet \
#     --encoder_path BAAI/bge-base-en


# python /home/mdafifal.mamun/research/S3M/src/main.py transformer \
#     --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json \
#     --bucket eclipse --trim_len 2 --lang java --multi_stack --loss ranknet \
#     --encoder_path BAAI/bge-base-en


# 20 frames
python /home/mdafifal.mamun/research/S3M/src/pretrain.py --bucket eclipse --trim_len 2 --max_frames 20
python /home/mdafifal.mamun/research/S3M/src/main.py transformer \
    --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json \
    --bucket eclipse --trim_len 2 --lang java --multi_stack --loss ranknet \
    --encoder_path /work/disa_lab/afif/projects/deduplication/trained_embedding_models/baai_bge-base-en_eclipse_20frames_4train_2trim_3850_traindays/final \
    # --skip_training

python /home/mdafifal.mamun/research/S3M/src/pretrain.py --bucket netbeans --trim_len 0 --max_frames 20
python /home/mdafifal.mamun/research/S3M/src/main.py transformer \
    --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/netbeans_2016/netbeans_stacktraces.json \
    --bucket_name netbeans --trim_len 0 --lang java --multi_stack --loss ranknet \
    --encoder_path /work/disa_lab/afif/projects/deduplication/trained_embedding_models/baai_bge-base-en_netbeans_20frames_4train_0trim_3850_traindays/final