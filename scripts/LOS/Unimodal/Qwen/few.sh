#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-08:59:59
#SBATCH --cpus-per-task=18
# Output and error files
#SBATCH -o final_outlogs/job.%J.out
#SBATCH -e final_errlogs/job.%J.err

# Set the Hugging Face cache directory to your scratch space
export HF_HOME="/scratch/baj321/.cache/huggingface"

# Set your Hugging Face authentication token
export HUGGING_FACE_HUB_TOKEN="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"
    
# Activating conda
eval "$(conda shell.bash hook)"
conda activate llama3_env

python agent_main.py \
    --data_path "/scratch/baj321/MedAgent/datasets/los_multimodal_dataset_splits/test.jsonl" \
    --few_shot_data_path "/scratch/baj321/MedAgent/datasets/los_multimodal_dataset_splits/train.jsonl" \
    --train_data_path "/scratch/baj321/MedAgent/datasets/los_multimodal_dataset_splits/train.jsonl" \
    --mimic_notes_dir "scratch/baj321/MIMIC-Note/physionet.org/files/mimic-iv-note/2.2/note/" \
    --task "los" \
    --modalities "ps" \
    --agent_setup "FewShot" \
    --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_dir "./results/los/Qwen/fewshot-ps" \
    --batch_size 3 \
    --num_shots 2 \
    --refine_iterations 1 \
    --debate_rounds 3 \
    --debug_samples 2