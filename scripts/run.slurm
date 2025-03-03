#!/bin/bash
#SBATCH -J s3m
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/research/S3M/out/train_out/gnome_multistack_3700days_%j.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracesim

# echo "Pretraining"
# python /home/mdafifal.mamun/research/S3M/src/pretrain.py

echo "Training"
# python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json --bucket eclipse --trim_len 2 --lang java --multi_stack --loss ranknet --encoder_path /home/mdafifal.mamun/research/S3M/models/bge-base-eclipse_10_class_prev_triplet/final
# python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json --bucket eclipse --trim_len 0 --lang java --multi_stack

# python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans --trim_len 0 --lang java --multi_stack

# python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans --trim_len 0 --lang java --multi_stack --loss triplet
# python /home/mdafifal.mamun/research/S3M/src/main.py transformer --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans --trim_len 0 --lang java --multi_stack --loss ranknet --encoder_path /work/disa_lab/afif/projects/deduplication/trained_embedding_models/baai_bge-base-en-v1.5_netbeans_10frames_5train_0trim_4200_traindays/final

# python /home/mdafifal.mamun/research/S3M/src/main.py s3m --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json --bucket_name eclipse --trim_len 2 --lang java --loss ranknet --encoder_path "ec_none"

python /home/mdafifal.mamun/research/S3M/src/main.py transformer --multi_stack --data_path /home/mdafifal.mamun/research/S3M/dataset/EMSE_data/gnome_2011/gnome_stacktraces_filtered.json --bucket_name gnome --trim_len 0 --lang cpp --loss ranknet --encoder_path /work/disa_lab/afif/projects/deduplication/trained_embedding_models/baai_bge-base-en-v1.5_gnome_100frames_1train_0trim_3700_traindays/final