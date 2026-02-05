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
export HF_HOME="/data/user/.cache/huggingface"

# Set your Hugging Face authentication token
export HUGGING_FACE_HUB_TOKEN="REDACTED_TOKEN"
    
# Activating conda
eval "$(conda shell.bash hook)"
conda activate llama3_env

python agent_main.py \
    --data_path "/data/user/MedAgent/datasets/multimodal_dataset_splits/test.jsonl" \
    --few_shot_data_path "/data/user/MedAgent/datasets/multimodal_dataset_splits/train.jsonl" \
    --modalities "ps" \
    --agent_setup "MAD" \
    --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_dir "./results/final_qwen/MAD-ps" \
    --batch_size 3 \
    --num_shots 2 \
    --refine_iterations 1 \
    --debate_rounds 3 \
    --debug_samples 2