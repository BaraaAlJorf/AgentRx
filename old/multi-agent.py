from __future__ import annotations

import os
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
import imp
import re

#Pipeline imports --------------------------------------------------------
from ehr_utils.preprocessing import Discretizer, Normalizer
from datasets.ehr_dataset import get_datasets
from datasets.cxr_dataset import get_cxr_datasets
from datasets.DataFusion import load_cxr_ehr_rr_dn
from arguments import args_parser
from pathlib import Path
from tqdm import tqdm
from tqdm import trange
import random
# NEW imports near the top
import io
import csv
import gc

# --------------------------------------------------------------------
# Hugging Face model setup
# --------------------------------------------------------------------
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    raise EnvironmentError("HUGGING_FACE_HUB_TOKEN environment variable not set.")

# model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
model_id = "HuggingFaceM4/idefics2-8b"


# Load processor and model
processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    token=hf_token,
    device_map="auto",  # automatically spreads layers across GPUs/CPU
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    offload_folder=f"/scratch/{os.getenv('USER', 'user')}/hf_offload"
)

# --------------------------------------------------------------------
# Prediction pipeline
# --------------------------------------------------------------------
# Define ETHNICITY mapping if missing
ETHNICITY = {'WHITE': 0,
 'UNKNOWN': 1,
 'OTHER': 2,
 'BLACK/AFRICAN AMERICAN': 3,
 'HISPANIC/LATINO': 4,
 'ASIAN': 5,
 'AMERICAN INDIAN/ALASKA NATIVE': 6,
 'UNABLE TO OBTAIN': 7}
 
# ---------- Compact EHR summarizer ----------
def ehr_compact_summary(ehr_sample: np.ndarray, max_features: int = 40, round_dp: int = 2) -> str:
    """
    Reduce (T, F) numeric timeseries to a short per-feature summary:
    - Select top `max_features` by variance (ignores NaNs)
    - For each selected feature return: mean, std, last, trend (slope)
    """
    x = np.asarray(ehr_sample)
    if x.ndim == 1:
        x = x[:, None]
    T, F = x.shape

    variances = np.zeros(F, dtype=float)
    for j in range(F):
        col = x[:, j]
        col = col[~np.isnan(col)]
        variances[j] = float(np.var(col)) if col.size > 0 else 0.0

    selected_idx = np.argsort(variances)[::-1][:min(max_features, F)]
    lines = []
    for j in selected_idx:
        col = x[:, j]
        col_nonan = col[~np.isnan(col)]
        if col_nonan.size == 0:
            continue
        mean = float(np.mean(col_nonan))
        std = float(np.std(col_nonan))
        last = float(col_nonan[-1])
        if col_nonan.size > 1:
            try:
                slope = float(np.polyfit(np.arange(len(col_nonan)), col_nonan, 1)[0])
            except Exception:
                slope = 0.0
        else:
            slope = 0.0
        lines.append(f"f{j}: mean={mean:.{round_dp}f}, std={std:.{round_dp}f}, last={last:.{round_dp}f}, trend={slope:.{round_dp}f}")
    return "\n".join(lines) if lines else "No numeric EHR data available."

def generate_response(batch_messages, processor, model, max_new_tokens=128):
    """
    Processes an entire BATCH of prompts. Images must be embedded in the messages.
    """
    inputs = processor.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        return_dict=True
    ).to(next(model.parameters()).device)

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                use_cache=False
            )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[ERROR] OOM during batched generation. Try reducing batch size.")
            return [""] * len(batch_messages)
        else:
            raise

    input_token_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[:, input_token_len:]
    decoded_outputs = processor.batch_decode(generated_tokens, skip_special_tokens=True)

    del inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return [text.strip() for text in decoded_outputs]

    # decode the model's newly generated text (after the prompt length)
    gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    decoded_output = processor.decode(gen_tokens, skip_special_tokens=True).strip()

    # cleanup to avoid accumulation
    try:
        del outputs
    except Exception:
        pass
    try:
        del inputs
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    return decoded_output

# ---------- Agent changes ----------
def run_cxr_agent(cxr_image, processor, model):
    prompt_text = (
        "You are a specialist radiologist. Analyze the provided chest X-ray. "
        "Summarize any critical findings that might indicate a high risk of mortality, "
        "such as widespread opacities, severe pleural effusion, or signs of ARDS. "
        "Provide a concise summary of your findings."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "content": cxr_image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    return generate_response(messages, processor, model, max_new_tokens=80)

def run_notes_agent(demographics_text, rad_note, discharge_note, processor, model):
    prompt_text = (
        "You are a clinical expert. Based on the following patient data, "
        "summarize the patient's condition and assess their stability. "
        "Focus on information relevant to their prognosis.\n\n"
        f"Patient Demographics:\n{demographics_text}\n\n"
        f"Radiology Note:\n{rad_note}\n\n"
        f"Discharge Note (if available):\n{discharge_note}\n\n"
        "Provide a concise summary."
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    return generate_response(messages, processor, model, max_new_tokens=80)

# NOTE: now accepts a compact textual summary instead of raw CSV
def run_ehr_agent(ehr_summary_text, processor, model):
    prompt_text = (
        "You are a clinical data analyst. Analyze the following compact EHR summary. "
        "Identify any significant trends that suggest clinical deterioration or stability. "
        "For example, look for trends in vitals, labs, or ventilator settings.\n\n"
        f"{ehr_summary_text}\n\n"
        "Provide a concise analysis of the trends."
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    return generate_response(messages, processor, model, max_new_tokens=80)

def run_coordinator_agent(cxr_summary, notes_summary, ehr_summary, processor, model):
    prompt_text = (
        "You are the lead physician responsible for the final decision. You have received "
        "reports from three specialists: Radiology, Clinical Notes, and EHR Data Analysis. "
        "Based *only* on the information in these reports, determine if the patient's "
        "in-hospital outcome was mortality.\n\n"
        f"Radiology Report:\n{cxr_summary}\n\n"
        f"Clinical Notes Summary:\n{notes_summary}\n\n"
        f"EHR Data Analysis:\n{ehr_summary}\n\n"
        "Task: Was the in-hospital outcome mortality? "
        "Respond with a single word only: 'Yes' or 'No'."
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    return generate_response(messages, processor, model, max_new_tokens=3)


def process_batch(batch, processor, model):
    ehr_data, cxr_data, discharge_notes, radiology_notes, targets_ehr, _, _, _, age, gender, ethnicity, hadm_id = batch
    
    batch_size = len(hadm_id)

    # --- Step 0: Prepare initial text inputs for the whole batch ---
    batch_demographics = [
        (f"- Age: {age[i].item()}\n"
         f"- Gender: {'Male' if gender[i].item() == 0 else 'Female'}\n"
         f"- Ethnicity: {list(ETHNICITY.keys())[list(ETHNICITY.values()).index(int(ethnicity[i].item()))]}")
        for i in range(batch_size)
    ]
    batch_ehr_summaries = [ehr_compact_summary(ehr_data[i], max_features=40) for i in range(batch_size)]

    # --- Step 1: Run CXR Agent for the entire batch in parallel ---
    cxr_agent_messages = [
        [{
            "role": "user",
            "content": [
                {"type": "image", "content": cxr_data[i]},
                {"type": "text", "text": (
                    "You are a specialist radiologist. Analyze the provided chest X-ray. "
                    "Summarize any critical findings that might indicate a high risk of mortality, "
                    "such as widespread opacities, severe pleural effusion, or signs of ARDS. "
                    "Provide a concise summary of your findings."
                )}
            ]
        }] for i in range(batch_size)
    ]
    batch_cxr_summaries = generate_response(cxr_agent_messages, processor, model, max_new_tokens=80)

    # --- Step 2: Run Notes Agent for the entire batch in parallel ---
    notes_agent_messages = [
        [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "You are a clinical expert. Based on the following patient data, "
                    "summarize the patient's condition and assess their stability. "
                    "Focus on information relevant to their prognosis.\n\n"
                    f"Patient Demographics:\n{batch_demographics[i]}\n\n"
                    f"Radiology Note:\n{radiology_notes[i]}\n\n"
                    f"Discharge Note (if available):\n{discharge_notes[i]}\n\n"
                    "Provide a concise summary."
                )
            }]
        }] for i in range(batch_size)
    ]
    batch_notes_summaries = generate_response(notes_agent_messages, processor, model, max_new_tokens=80)

    # --- Step 3: Run EHR Agent for the entire batch in parallel ---
    ehr_agent_messages = [
        [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "You are a clinical data analyst. Analyze the following compact EHR summary. "
                    "Identify any significant trends that suggest clinical deterioration or stability. "
                    "For example, look for trends in vitals, labs, or ventilator settings.\n\n"
                    f"{batch_ehr_summaries[i]}\n\n"
                    "Provide a concise analysis of the trends."
                )
            }]
        }] for i in range(batch_size)
    ]
    batch_ehr_analyses = generate_response(ehr_agent_messages, processor, model, max_new_tokens=80)

    # --- Step 4: Run Coordinator Agent for the entire batch in parallel ---
    coordinator_agent_messages = [
        [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "You are the lead physician responsible for the final decision. You have received "
                    "reports from three specialists: Radiology, Clinical Notes, and EHR Data Analysis. "
                    "Based *only* on the information in these reports, determine if the patient's "
                    "in-hospital outcome was mortality.\n\n"
                    f"Radiology Report:\n{batch_cxr_summaries[i]}\n\n"
                    f"Clinical Notes Summary:\n{batch_notes_summaries[i]}\n\n"
                    f"EHR Data Analysis:\n{batch_ehr_analyses[i]}\n\n"
                    "Task: Was the in-hospital outcome mortality? "
                    "Respond with a single word only: 'Yes' or 'No'."
                )
            }]
        }] for i in range(batch_size)
    ]
    batch_final_decisions = generate_response(coordinator_agent_messages, processor, model, max_new_tokens=3)
    
    # --- Step 5: Process final decisions ---
    predictions = []
    for decision in batch_final_decisions:
        text = decision.lower()
        if "yes" in text:
            predictions.append(1)
        elif "no" in text:
            predictions.append(0)
        else:
            predictions.append(-1)
            
    true_labels = [int(t) for t in targets_ehr]
    
    return predictions, true_labels


def calculate_accuracy(predictions, true_labels):
    correct, total = 0, 0
    for pred, true in zip(predictions, true_labels):
        if pred in [0, 1]:
            total += 1
            if pred == true:
                correct += 1
    return (correct / total) * 100 if total > 0 else 0.0


# --------------------------------------------------------------------
# Load datasets
# --------------------------------------------------------------------
args = args_parser().parse_args()
path = Path(args.save_dir)
path.mkdir(parents=True, exist_ok=True)

seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

# Set PyTorch to deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def read_timeseries(args):
    path = f'{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)
    

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')


discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

ehr_train, ehr_val, ehr_test = get_datasets(discretizer, normalizer, args)
cxr_train, cxr_val, cxr_test = get_cxr_datasets(args)

train_dl, val_dl, test_dl = load_cxr_ehr_rr_dn(
    args, ehr_train, ehr_val, cxr_train, cxr_val, ehr_test, cxr_test
)

# --------------------------------------------------------------------
# Run evaluation
# --------------------------------------------------------------------
all_predictions, all_true_labels = [], []

for i, batch in enumerate(tqdm(test_dl, desc="Evaluating", unit="batch")):
    batch_predictions, batch_true_labels = process_batch(batch, processor, model)
    all_predictions.extend(batch_predictions)
    all_true_labels.extend(batch_true_labels)
    # if i == 2:  # Stop after the second batch
    #     break

accuracy = calculate_accuracy(all_predictions, all_true_labels)
print(f"Total Accuracy: {accuracy:.2f}%")