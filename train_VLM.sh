#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-08:59:59
#SBATCH --cpus-per-task=18
#SBATCH -o outlogs/job.%J.out
#SBATCH -e errlogs/job.%J.err

export HF_HOME="/scratch/baj321/.cache/huggingface"
export HUGGING_FACE_HUB_TOKEN="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"
export HF_HUB_ENABLE_DOWNLOAD_PROGRESS=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module purge
module load cuda/11.8.0
module load gcc/9.2.0

eval "$(conda shell.bash hook)"
conda activate llama3_env

python train_vlm.py \
    --model_id "OpenGVLab/InternVL2-8B" \
    --train_data_path "/scratch/baj321/MedAgent/datasets/multimodal_dataset_splits/train.jsonl" \
    --test_data_path "/scratch/baj321/MedAgent/datasets/multimodal_dataset_splits/test.jsonl" \
    --output_dir "./Intern/Multimodal/finetuned_models" \
    --lora_adapter_path "/scratch/baj321/MedAgent/finetuned_models/checkpoint-epoch-3/" \
    --grad_accum_steps 8 \
    --batch_size 3 \
    --mode train \
    --epochs 3 \
    --modalities "ps"