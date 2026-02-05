#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-08:59:59
#SBATCH --cpus-per-task=18
#SBATCH -o outlogs/job.%J.out
#SBATCH -e errlogs/job.%J.err

export HF_HOME=".cache/huggingface"
export HUGGING_FACE_HUB_TOKEN=""
export HF_HUB_ENABLE_DOWNLOAD_PROGRESS=1
export HF_HUB_ENABLE_HF_TRANSFER=1

eval "$(conda shell.bash hook)"
conda activate llama3_env

# rm -rf "${HF_HOME}/hub"  # Only if you're sure!

python - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"

snapshot_download(
    repo_id=repo_id,
    revision="main",          # optional: or a specific commit SHA
    resume_download=True      # resumes if job preempts; verifies checksums
)
print("Prefetch complete.")
PY
