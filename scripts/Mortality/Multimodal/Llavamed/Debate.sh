#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-08:59:59
#SBATCH --cpus-per-task=18
# Output and error files
#SBATCH -o final_outlogs/job.%J.out
#SBATCH -e final_errlogs/job.%J.err

# 1. Clear previous modules to avoid conflicts
module purge

# 2. Load the compiler (THIS IS THE FIX)
module load cuda/11.8.0
module load gcc/9.2.0

# Set the Hugging Face cache directory to your scratch space
export HF_HOME="/scratch/baj321/.cache/huggingface"

# Set your Hugging Face authentication token
export HUGGING_FACE_HUB_TOKEN="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"
    
# Activating conda
eval "$(conda shell.bash hook)"
conda activate llama3_env

python agent_main.py \
    --data_path "/scratch/baj321/MedAgent/datasets/multimodal_dataset_splits/test.jsonl" \
    --few_shot_data_path "/scratch/baj321/MedAgent/datasets/multimodal_dataset_splits/train.jsonl" \
    --modalities "ps-cxr-ehr-rr" \
    --agent_setup "Debate_Multimodal" \
    --model_id "chaoyinshe/llava-med-v1.5-mistral-7b-h" \
    --output_dir "./results/Llava/debate-full" \
    --batch_size 1 \
    --num_shots 2 \
    --refine_iterations 1 \
    --debate_rounds 3 \
    --debug_samples 2