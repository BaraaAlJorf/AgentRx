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
    inputs = processor.apply_chat_template(
        batch_messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", padding=True, return_dict=True
    ).to(next(model.parameters()).device)

    try:
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[ERROR] OOM during batched generation.")
            return ["ERROR: OOM"] * len(batch_messages)
        else: raise

    input_token_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[:, input_token_len:]
    decoded_outputs = processor.batch_decode(generated_tokens, skip_special_tokens=True)

    del inputs, outputs, generated_tokens
    gc.collect()
    torch.cuda.empty_cache()

    return [text.strip() for text in decoded_outputs]


def process_batch_with_debate(batch, processor, model, num_rounds=3):
    ehr_data, cxr_data, discharge_notes, radiology_notes, targets_ehr, _, _, _, age, gender, ethnicity, _ = batch
    batch_size = len(age)

    # --- Step 0: Create a complete "Case File" for each patient ---
    batch_case_files = []
    for i in range(batch_size):
        case_file = (
            f"--- Patient Case File ---\n"
            f"**Demographics**:\n- Age: {age[i].item()}\n- Gender: {'Male' if gender[i].item() == 0 else 'Female'}\n"
            f"- Ethnicity: {list(ETHNICITY.keys())[list(ETHNICITY.values()).index(int(ethnicity[i].item()))]}\n\n"
            f"**Clinical Notes**:\n- Radiology Note: {radiology_notes[i]}\n- Discharge Note: {discharge_notes[i]}\n\n"
            ## MODIFIED ## - Uses the single ehr_compact_summary function
            f"**EHR Time-Series Summary**:\n{ehr_compact_summary(ehr_data[i])}\n"
            f"-------------------------"
        )
        batch_case_files.append(case_file)

    batch_debate_history: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
    DEBATING_AGENTS = ["Agent A", "Agent B", "Agent C"]

    # --- Step 1: The 3-Round Debate ---
    for round_num in range(num_rounds):
        all_agents_messages_this_round = []
        for i in range(batch_size):
            for agent_name in DEBATING_AGENTS:
                other_agents_opinions = ""
                if round_num > 0:
                    other_agents_opinions = f"In round {round_num}, your colleagues reported:\n"
                    last_round_reports = [r for r in batch_debate_history[i] if r['round'] == round_num]
                    for report in last_round_reports:
                        if report['agent'] != agent_name:
                            other_agents_opinions += f"- {report['agent']}: Prediction was '{report['prediction']}'. Analysis: {report['analysis']}\n"
                    other_agents_opinions += "\nRe-evaluate the case based on their views and the full data.\n"

                ## MODIFIED ## - Prompt now asks for more detailed text in the debate.
                prompt_text = (
                    f"You are {agent_name}, a clinical AI expert tasked with predicting in-hospital mortality. "
                    f"Analyze the entire patient case file provided.\n\n"
                    f"{other_agents_opinions}"
                    f"Provide a brief analysis (2-3 sentences) outlining your reasoning, focusing on the most critical factors. "
                    f"Then, on a new line, state your prediction as a single word: 'Yes' for mortality or 'No'."
                    f"\n\n{batch_case_files[i]}"
                )
                
                all_agents_messages_this_round.append(
                    [{"role": "user", "content": [{"type": "image", "content": cxr_data[i]}, {"type": "text", "text": prompt_text}]}]
                )
        
        ## MODIFIED ## - Increased tokens to allow for more detailed debate text.
        all_responses = generate_response(all_agents_messages_this_round, processor, model, max_new_tokens=120)
        
        response_idx = 0
        for i in range(batch_size):
            for agent_name in DEBATING_AGENTS:
                full_response = all_responses[response_idx]
                response_idx += 1
                
                lines = full_response.strip().split('\n')
                analysis = lines[0]
                prediction = "Unknown"
                if len(lines) > 1:
                    last_word = lines[-1].lower().strip(" .")
                    if "yes" in last_word: prediction = "Yes"
                    elif "no" in last_word: prediction = "No"

                batch_debate_history[i].append({
                    "round": round_num + 1, "agent": agent_name,
                    "analysis": analysis, "prediction": prediction
                })

    # --- Step 2: The Coordinator's Final Decision ---
    coordinator_messages = []
    for i in range(batch_size):
        transcript = f"--- DEBATE TRANSCRIPT ---\n"
        for round_num in range(1, num_rounds + 1):
            transcript += f"\n--- Round {round_num} ---\n"
            round_reports = [r for r in batch_debate_history[i] if r['round'] == round_num]
            for report in round_reports:
                transcript += f"{report['agent']} predicted '{report['prediction']}': \"{report['analysis']}\"\n"
        
        coordinator_prompt = (
            "You are the Lead Physician. Based *only* on the following debate between three clinical AI agents, "
            "make the final determination of in-hospital mortality. Synthesize their arguments to make your decision.\n\n"
            f"{transcript}\n\n"
            "Task: What is the final outcome? Respond with a single word only: 'Yes' or 'No'."
        )
        coordinator_messages.append([{"role": "user", "content": [{"type": "text", "text": coordinator_prompt}]}])

    final_decisions = generate_response(coordinator_messages, processor, model, max_new_tokens=3)

    # --- Step 3: Parse Final Predictions ---
    predictions = []
    for decision in final_decisions:
        text = decision.lower()
        if "yes" in text: predictions.append(1)
        elif "no" in text: predictions.append(0)
        else: predictions.append(-1)
            
    true_labels = [int(t) for t in targets_ehr]
    return predictions, true_labels


def calculate_accuracy(predictions, true_labels):
    valid_preds = [(p, t) for p, t in zip(predictions, true_labels) if p != -1]
    if not valid_preds: return 0.0
    correct = sum(1 for p, t in valid_preds if p == t)
    return (correct / len(valid_preds)) * 100


args = args_parser().parse_args()
path = Path(args.save_dir)
path.mkdir(parents=True, exist_ok=True)

seed = 1002
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def read_timeseries(args): # Dummy function placeholder
    path = f'{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv'
    ret = [];
    with open(path, "r") as tsfile:
        tsfile.readline()
        for line in tsfile: ret.append(np.array(line.strip().split(',')))
    return np.stack(ret)

discretizer = Discretizer(timestep=float(args.timestep), store_masks=True, impute_strategy='previous', start_time='zero')
discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if "->" not in x]
normalizer = Normalizer(fields=cont_channels)
normalizer_state = args.normalizer_state or 'normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
normalizer.load_params(os.path.join(os.path.dirname(__file__), normalizer_state))

ehr_train, ehr_val, ehr_test = get_datasets(discretizer, normalizer, args)
cxr_train, cxr_val, cxr_test = get_cxr_datasets(args)
_, _, test_dl = load_cxr_ehr_rr_dn(args, ehr_train, ehr_val, cxr_train, cxr_val, ehr_test, cxr_test)

all_predictions, all_true_labels = [], []
for i, batch in enumerate(tqdm(test_dl, desc="Evaluating Batches", unit="batch")):
    batch_predictions, batch_true_labels = process_batch_with_debate(batch, processor, model, num_rounds=3)
    all_predictions.extend(batch_predictions)
    all_true_labels.extend(batch_true_labels)

accuracy = calculate_accuracy(all_predictions, all_true_labels)
print(f"\nFinal Accuracy from Coordinator: {accuracy:.2f}%")