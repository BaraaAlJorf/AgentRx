# llama3_mortality_prompting.py
"""End‑to‑end mortality prediction with Meta‑Llama‑3‑70B/8B (text‑only).

This **single script**:

* Parses CLI flags via your existing *arguments.py* (`args_parser()`)
* Re‑creates the full data pipeline (EHR + rad & discharge notes)
* Prompts Llama‑3 **once per patient** ( ⩤ no batching inside the LM ⩥)
* Computes **AUROC** and **AUPRC** on the test set
* Saves `predictions.csv` & `metrics.txt` under `args.save_dir`

**Modalities included**: EHR time‑series, radiology report, discharge summary.
(CXR tensors are still loaded for completeness but *ignored* in the prompt for
now – flip the TODO flag later when you want them.)
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, average_precision_score

# 🏥  Pipeline imports --------------------------------------------------------
from ehr_utils.preprocessing import Discretizer, Normalizer
from datasets.ehr_dataset import get_datasets
from datasets.cxr_dataset import get_cxr_datasets  # still built but ignored
from datasets.DataFusion import load_cxr_ehr_rr_dn
from arguments import args_parser

# ---------------------------------------------------------------------------
# ⚙️ Llama‑3 config – tweak via env‑vars or constants
# ---------------------------------------------------------------------------
MODEL_ID = os.getenv("LLAMA3_MODEL", "meta-llama/Meta-Llama-3-70B")
HF_TOKEN = os.getenv("HF_TOKEN", "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")  # ⬅️  replace with real
RAW_MODE = bool(int(os.getenv("RAW_MODE", "0")))     # 1 → raw EHR, 0 → summary
MAX_EHR_TIMESTEPS = int(os.getenv("MAX_EHR_STEPS", "48"))  # crop for context

# ---------------------------------------------------------------------------
# 📝 Parse runtime arguments (no __main__ guard – runs on import)
# ---------------------------------------------------------------------------
args = args_parser().parse_args()
print("Runtime arguments →")
for k, v in vars(args).items():
    print(f"  {k}: {v}")

SAVE_DIR = Path(args.save_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 🔄  Reproducibility
# ---------------------------------------------------------------------------
seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# 📦 Model loading
# ---------------------------------------------------------------------------
print(f"\nLoading {MODEL_ID} …  (RAW_MODE={RAW_MODE})")

_tokenizer_kwargs = dict(use_auth_token=HF_TOKEN)
model_kwargs = dict(device_map="auto", torch_dtype=torch.float16, use_auth_token=HF_TOKEN)

if "gptq" in MODEL_ID.lower():  # e.g. decapoda‑research
    model_kwargs.pop("torch_dtype", None)

print(" > tokenizer …", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **_tokenizer_kwargs)

print(" > model …", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
model.eval()

# ---------------------------------------------------------------------------
# 🏗️ Data – build discretiser, normaliser, datasets & loaders
# ---------------------------------------------------------------------------
print("\nBuilding discretiser / normaliser …", flush=True)

discretizer = Discretizer(
    timestep=float(args.timestep),
    store_masks=True,
    impute_strategy="previous",
    start_time="zero",
)

# Use a single sample to grab header names
sample_ts_path = f"{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv"
if not Path(sample_ts_path).exists():
    raise FileNotFoundError(f"Sample timeseries CSV not found: {sample_ts_path}")

def _read_timeseries(fp: str) -> np.ndarray:
    """Read a single timeseries CSV → ndarray[T, F+1] (Hours + features)."""
    with open(fp, "r") as f:
        header = f.readline().strip().split(",")
        assert header[0] == "Hours"
        rows = [np.array(line.strip().split(","), dtype="object") for line in f]
    return np.stack(rows)

# Header list (skip "Hours")
discretizer_header = discretizer.transform(_read_timeseries(sample_ts_path))[1].split(",")[1:]

cont_channels = [i for i, col in enumerate(discretizer_header) if "->" not in col]
normalizer = Normalizer(fields=cont_channels)

norm_state = args.normalizer_state or f"normalizers/ph_ts{args.timestep}.input_str:previous.start_time:zero.normalizer"
normalizer.load_params(norm_state)

print("Data → datasets & loaders …", flush=True)

ehr_train, ehr_val, ehr_test = get_datasets(discretizer, normalizer, args)

cxr_train, cxr_val, cxr_test = get_cxr_datasets(args)

train_dl, val_dl, test_dl = load_cxr_ehr_rr_dn(
    args,
    ehr_train,
    ehr_val,
    cxr_train,
    cxr_val,
    ehr_test,
    cxr_test,
)

# ---------------------------------------------------------------------------
# 📝 Prompt helpers
# ---------------------------------------------------------------------------

def _as_tensor(arr) -> torch.Tensor:
    return arr if torch.is_tensor(arr) else torch.as_tensor(arr)


def _ehr_to_csv(ehr_np: np.ndarray, header: List[str]) -> str:
    """Convert `[T, F]` numpy array → CSV (limited to MAX_EHR_TIMESTEPS)."""
    t = min(ehr_np.shape[0], MAX_EHR_TIMESTEPS)
    buf = io.StringIO()
    print(",".join(["Hour"] + header), file=buf)
    for hr in range(t):
        row = [f"{hr}"] + [f"{x:.3f}" if not np.isnan(x) else "" for x in ehr_np[hr]]
        print(",".join(row), file=buf)
    return buf.getvalue()


def summarise_ehr(ehr_np: np.ndarray, header: List[str]) -> str:
    """Return compact summary – last observed value per channel."""
    mask = ~np.isnan(ehr_np)
    last_idx = len(ehr_np) - 1 - mask[::-1].argmax(axis=0)
    last_vals = ehr_np[last_idx, np.arange(ehr_np.shape[1])]
    pairs = [f"{feat}={val:.2f}" for feat, val in zip(header, last_vals)]
    return ", ".join(pairs)


def build_prompt_single(ehr_np: np.ndarray, discharge: str, radiology: str) -> str:
    """Build prompt for **one** patient."""
    if RAW_MODE:
        ehr_block = _ehr_to_csv(ehr_np, discretizer_header)
        prompt = (
            "You are a clinical decision support AI. Below are raw patient data. "
            "Predict in-hospital mortality (`0` survive, `1` die).\n\n"
            "== Discretised EHR CSV ==\n" + ehr_block + "\n"
            "== Radiology Report ==\n" + radiology.strip() + "\n\n"
            "== Discharge Summary ==\n" + discharge.strip() + "\n\n"
            "Answer with a single character: 0 or 1. Nothing else."
        )
    else:
        prompt = (
            "You are a clinical decision support AI. Based on the patient information below, "
            "predict in-hospital mortality.\n\n"
            "== Structured EHR (last values) ==\n" + summarise_ehr(ehr_np, discretizer_header) + "\n\n"
            "== Radiology Report ==\n" + radiology.strip() + "\n\n"
            "== Discharge Summary ==\n" + discharge.strip() + "\n\n"
            "Answer with a single character: `0` if the patient survives, `1` if the patient dies. Do NOT output anything else."
        )
    return prompt

# ---------------------------------------------------------------------------
# 🤖 Generation loop with auto‑reprompting (single patient)
# ---------------------------------------------------------------------------

def _extract_answer(text: str) -> str:
    m = re.search(r"[01]", text)
    return m.group(0) if m else ""


def generate_label(prompt: str, max_new_tokens: int = 4, attempts: int = 5) -> Optional[int]:
    """Call Llama‑3 until it returns 0/1 or we hit `attempts`."""
    for _ in range(attempts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = _extract_answer(full[len(prompt):])
        if answer in {"0", "1"}:
            return int(answer)
        prompt += "\nYour answer was invalid. Reply with only 0 or 1."
    return None

# ---------------------------------------------------------------------------
# 🚀 Inference across the **test** loader
# ---------------------------------------------------------------------------
print("\nRunning inference on *test* set …", flush=True)

y_true, y_pred, csv_rows = [], [], ["patient_id,truth,pred"]

for batch_idx, batch in enumerate(test_dl, 1):
    # Unpack collate output (see DataFusion.my_collate)
    ehr_np        = batch[0]            # ndarray [B, T, F]
    discharge_lst = batch[2]            # list[str] len=B
    radiology_lst = batch[3]            # list[str]
    targets_ehr   = batch[4]            # np.ndarray shape [B] or [B,1]
    hadm_ids      = batch[11]           # Tensor [B]

    batch_size = len(discharge_lst)

    for i in range(batch_size):
        ehr_i = np.asarray(ehr_np[i])
        discharge_i = discharge_lst[i] or ""
        radiology_i = radiology_lst[i] or ""
        truth_i = int(targets_ehr[i]) if np.ndim(targets_ehr) == 1 else int(targets_ehr[i][0])
        pid_i = int(hadm_ids[i].item()) if torch.is_tensor(hadm_ids) else hadm_ids[i]

        prompt = build_prompt_single(ehr_i, discharge_i, radiology_i)
        pred_i = generate_label(prompt)

        csv_rows.append(f"{pid_i},{truth_i},{'' if pred_i is None else pred_i}")

        if pred_i is not None:
            y_true.append(truth_i)
            y_pred.append(pred_i)

    if batch_idx % 10 == 0:
        print(f" … batch {batch_idx} done (samples so far: {len(y_true)})", flush=True)

# ---------------------------------------------------------------------------
# 📊  Metrics & saving
# ---------------------------------------------------------------------------
print("\nComputing AUROC / AUPRC …", flush=True)
if len(set(y_true)) == 2:
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    metric_text = f"AUROC = {auroc:.4f}\nAUPRC = {auprc:.4f}\n"
else:
    metric_text = "AUROC/AUPRC undefined – only one class in ground truth or predictions.\n"
print(metric_text)

pred_path = SAVE_DIR / "predictions.csv"
metric_path = SAVE_DIR / "metrics.txt"

pred_path.write_text("\n".join(csv_rows))
metric_path.write_text(metric_text)

print(f"✅  Saved predictions → {pred_path.resolve()}")
print(f"✅  Saved metrics     → {metric_path.resolve()}")
