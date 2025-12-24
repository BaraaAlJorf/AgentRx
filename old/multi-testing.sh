#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-08:59:59
#SBATCH --cpus-per-task=18
# Output and error files
#SBATCH -o outlogs/job.%J.out
#SBATCH -e errlogs/job.%J.err

# Set the Hugging Face cache directory to your scratch space
export HF_HOME="/scratch/baj321/.cache/huggingface"

# Set your Hugging Face authentication token
export HUGGING_FACE_HUB_TOKEN="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"
    
# Activating conda
eval "$(conda shell.bash hook)"
conda activate llama3_env

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python multi-agent.py \
--mode train \
--epochs 1 --batch_size 8 --lr 0.001 \
--num_classes 1 \
--modalities EHR-CXR-RR \
--text_encoder biobert \
--classifier mlp \
--loss bce \
--save_dir 'checkpoints/calibrated' \
--task in-hospital-mortality \
--labels_set mortality \
--output_dim 512 \
--output_dim_cxr 512 \
--output_dim_rr 512 \
--data_pairs paired \