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

module purge
module load cuda/11.8.0
module load gcc/9.2.0

eval "$(conda shell.bash hook)"
conda activate llama3_env

pip uninstall -y flash-attn

echo "Starting Flash Attention build..."
echo "Environment: $CONDA_PREFIX"
echo "CUDA Version: $(nvcc --version | grep release)"
pip install ninja packaging
echo "Ninja location: $(which ninja)"
echo "Ninja version: $(ninja --version)"

export MAX_JOBS=4

pip install flash-attn --no-build-isolation --no-cache-dir -vv

echo "Installation finished"