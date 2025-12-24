import os
import gc
import re # NEW: For parsing probabilities
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# --------------------------------------------------------------------
# 🧠 1. Model & Processor Setup (Lazy Loading)
# --------------------------------------------------------------------
MODEL_CACHE = {}

def get_model_and_processor(args):
    """Loads and caches the specified Hugging Face model and processor."""
    model_id = args.model_id
    if model_id in MODEL_CACHE:
        return MODEL_CACHE[model_id]

    print(f"Initializing Model and Processor for '{model_id}'...")
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise EnvironmentError("HUGGING_FACE_HUB_TOKEN environment variable not set.")

    try:
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, token=hf_token, device_map="auto",
            low_cpu_mem_usage=True, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        
        # REMOVED: yes_id and no_id logic is no longer needed

        print(f"✅ Model '{model_id}' and Processor loaded successfully.")
        MODEL_CACHE[model_id] = (model, processor) # Store model and processor
        return model, processor
    except Exception as e:
        print(f"❌ Failed to load model '{model_id}'. Error: {e}")
        exit()

# --------------------------------------------------------------------
# 🛠️ 2. Shared Helper Functions
# --------------------------------------------------------------------

# NEW: Regex to find the first valid number (e.g., 0.12, .12, 0)
PROB_REGEX = re.compile(r"(\d*\.?\d+)") 

def _parse_probability_string(text: str) -> float:
    """Safely parses a string to find the first valid probability."""
    match = PROB_REGEX.search(text)
    if match:
        try:
            # Get first matched group, convert to float, and clip to [0, 1]
            prob = float(match.group(1))
            return np.clip(prob, 0.0, 1.0) 
        except ValueError:
            return 0.5 # Default on parsing error (e.g., "1.2.3")
    return 0.5 # Default if no number found

def generate_response(prompts, model, processor, max_tokens):
    """
    (UNCHANGED)
    Processes a batch of prompts for free-text generation (e.g., specialist analysis).
    """
    inputs = processor.apply_chat_template(
        prompts,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        return_dict=True
    ).to(model.device)

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=1,
                use_cache=False
            )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[ERROR] OOM during batched generation. Try reducing batch size.")
            return [""] * len(prompts)
        raise

    input_token_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[:, input_token_len:]
    cleaned_outputs = processor.batch_decode(generated_tokens, skip_special_tokens=True)

    del inputs, outputs, generated_tokens
    
    return [text.strip() for text in cleaned_outputs]
    

def generate_probability_with_scores(prompts, model, processor, max_tokens=10):
    """
    NEW FUNCTION: Replaces generate_decision_with_scores
    
    Generates a probability string (e.g., "0.35") and calculates two scores:
    1.  Explicit Probability: The float value parsed from the string.
    2.  Implicit Probability: The model's average confidence (probability) 
        in generating the tokens that form the probability string.
    
    Returns:
        texts (list[str]): The raw generated text (e.g., "0.35")
        explicit_probs (np.array): The parsed float probabilities.
        implicit_probs (np.array): The avg token probabilities for the number.
    """
    inputs = processor.apply_chat_template(
        prompts,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        return_dict=True
    ).to(model.device)

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=1,
                use_cache=False,
                return_dict_in_generate=True, # MUST be True
                output_scores=True           # MUST be True
            )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[ERROR] OOM during batched generation. Try reducing batch size.")
            dummy_probs = np.full(len(prompts), 0.5)
            dummy_texts = ["0.5"] * len(prompts)
            return dummy_texts, dummy_probs, dummy_probs
        raise

    input_token_len = inputs["input_ids"].shape[1]
    # [batch_size, num_generated_tokens]
    generated_tokens = outputs.sequences[:, input_token_len:]
    
    # --- Calculate Implicit Probability ---
    
    # `outputs.scores` is a tuple of tensors, one for each generated token step
    # Each tensor is [batch_size, vocab_size]
    
    # 1. Stack the scores: [num_tokens, batch_size, vocab_size]
    all_token_logits = torch.stack(outputs.scores, dim=0)
    # 2. Permute to: [batch_size, num_tokens, vocab_size]
    all_token_logits = all_token_logits.permute(1, 0, 2)
    # 3. Get probabilities for all tokens in vocab
    all_token_probs = F.softmax(all_token_logits, dim=-1)
    
    # 4. Get the probs of the *chosen* tokens.
    # `generated_tokens` is [batch_size, num_tokens]
    # We use `gather` to pick the prob of the token that was actually generated
    # `chosen_token_probs` will be [batch_size, num_tokens]
    chosen_token_probs = torch.gather(
        all_token_probs, 
        dim=2, 
        index=generated_tokens.unsqueeze(-1)
    ).squeeze(-1)

    # 5. Decode text and calculate final scores
    cleaned_texts = processor.batch_decode(generated_tokens, skip_special_tokens=True)
    
    final_explicit_probs = []
    final_implicit_probs = []
    final_cleaned_texts = []
    
    # ==================== FIX IS HERE ====================
    # Access pad_token_id via the tokenizer
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        # Fallback to eos_token_id if pad_token_id is not set
        pad_token_id = processor.tokenizer.eos_token_id
    # ================= END OF FIX ======================

    for i in range(len(prompts)):
        text = cleaned_texts[i].strip()
        final_cleaned_texts.append(text)
        
        # 6. Parse Explicit Probability
        explicit_prob = _parse_probability_string(text)
        final_explicit_probs.append(explicit_prob)
        
        # 7. Calculate Implicit Probability
        # Get the token probabilities for the i-th sequence
        seq_token_probs = chosen_token_probs[i] # Shape: [num_tokens]
        
        # Find non-padding tokens
        actual_tokens = generated_tokens[i]
        non_pad_mask = (actual_tokens != pad_token_id)
        
        if non_pad_mask.sum() == 0:
            # Empty generation, default to 0.5
            final_implicit_probs.append(0.5)
        else:
            # Select only the probabilities of non-padding tokens
            valid_token_probs = seq_token_probs[non_pad_mask]
            # Calculate the mean probability
            mean_prob = valid_token_probs.mean().item()
            final_implicit_probs.append(mean_prob)

    del inputs, outputs, all_token_logits, all_token_probs, chosen_token_probs, generated_tokens
    
    return (
        final_cleaned_texts, 
        np.array(final_explicit_probs, dtype=np.float32), 
        np.array(final_implicit_probs, dtype=np.float32)
    )


def format_ehr_as_text(path: str, max_features: int = 15) -> str:
    if not os.path.exists(path):
        return "EHR timeseries data not available."
    try:
        x = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(1, 77))
    except Exception as e:
        return f"Could not process EHR data: {e}"
    if x.ndim == 1: x = x[:, None]
    variances = np.nanvar(x, axis=0)
    selected_idx = np.argsort(variances)[::-1][:min(max_features, x.shape[1])]
    lines = ["Recent EHR Vitals and Labs (most recent values last):"]
    for j in selected_idx:
        col_nonan = x[:, j][~np.isnan(x[:, j])]
        if col_nonan.size == 0: continue
        recent_values = col_nonan[-5:]
        values_str = ", ".join([f"{v:.1f}" for v in recent_values])
        lines.append(f"- Feature {j}: [{values_str}]")
    return "\n".join(lines) if len(lines) > 1 else "No valid numeric EHR data."


def _format_batch_results(batch, gen_texts, explicit_probs, implicit_probs, modality_requests_list):
    """
    MODIFIED: Formats results to include explicit and implicit probs.
    """
    batch_results = []
    for i, patient_data in enumerate(batch):
        phenotype_labels = [k for k in patient_data['labels'].keys() if k != 'in_hospital_mortality_48hr']
        pheno_preds = {label: -1 for label in phenotype_labels}

        # Get the explicit probability and derived prediction
        exp_prob = float(explicit_probs[i])
        # Binary prediction is now derived from the explicit probability
        prediction = 1 if exp_prob > 0.5 else 0 
        
        result = {
            'stay_id': patient_data['stay_id'], 'ground_truth': patient_data['labels'],
            'predictions': {
                'in_hospital_mortality_48hr': int(prediction), 
                'phenotypes': pheno_preds,
                # NEW: Store all probability info
                'mortality_probability': exp_prob, # This is the explicit (parsed) prob
                'mortality_probability_implicit': float(implicit_probs[i]), # This is the implicit (avg token) prob
                'mortality_probability_text': gen_texts[i] # The raw text output
            },
            'modality_requests': modality_requests_list[i],
            'modality_availability': {
                'cxr': patient_data['cxr_image_path'] != 'CXR image not yet available (no valid image in first 48 hours).',
                'radiology_report': patient_data['radiology_report_text'] != 'Radiology report not yet available (no report in first 48 hours).'
            }
        }
        batch_results.append(result)
    return batch_results

# --------------------------------------------------------------------
# 🤖 3. Agent Architecture Implementations
# --------------------------------------------------------------------

def _build_prompt_content(patient_data, ehr_text):
    
    content = []
    has_image = False
    if patient_data['cxr_image_path'] != 'CXR image not yet available (no valid image in first 48 hours).':
        if os.path.exists(patient_data['cxr_image_path']):
            try:
                content.append({"type": "image", "content": Image.open(patient_data['cxr_image_path']).convert("RGB")})
                has_image = True
            except: pass
    
    # MODIFIED: Changed prompt to ask for probability
    prompt_text = (
        "You are an expert ICU risk prediction model."
        " Analyze the following data collected during the first 48 hours in the intensive care unit to determine the probability of in-hospital mortality."
        " Note that some data modalities may be missing, so only process the ones that are provided."
        " Respond *only* with the probability as a decimal number (e.g., '0.35'). Do not add any other text.\n\n"
        "--- DATA ---\n"
        f"Patient Summary:\n{patient_data['patient_summary_text']}\n\n"
        f"Radiology Report:\n{patient_data['radiology_report_text']}\n\n"
        f"EHR:\n{ehr_text}\n\n"
        "--- PROBABILITY ---"
    )
    content.append({"type": "text", "text": prompt_text})
    return content, has_image

def _run_single_agent_batch(batch, model, processor, args):
    """MODIFIED: Implements the "SingleAgent" approach."""
    prompts = []
    modality_requests = []
    for patient in batch:
        ehr_text = format_ehr_as_text(patient['ehr_timeseries_path'])
        content, has_image = _build_prompt_content(patient, ehr_text)
        
        prompts.append([{"role": "user", "content": content}])
        modality_requests.append({
            'patient_summary': 1, 'ehr_timeseries': 1, 'radiology_report': 1, 'cxr': 1 if has_image else 0
        })

    # MODIFIED: Call new function
    texts, exp_probs, imp_probs = generate_probability_with_scores(
        prompts, model, processor, max_tokens=10
    )
    return _format_batch_results(batch, texts, exp_probs, imp_probs, modality_requests)

def _run_multi_agent_batch(batch, model, processor, args):
    """MODIFIED: Implements the "MultiAgent" (MedCollab) approach."""
    max_tokens = args.max_new_tokens
    
    # --- Specialist Analyses (Unchanged) ---
    # EHR Agent
    prompts = [[{"role": "user", "content": [{"type": "text", "text": f"You are a clinical data analyst. Based on the following data, what is the patient's stability trend?\n\n{format_ehr_as_text(p['ehr_timeseries_path'])}\n\nProvide a concise one-sentence analysis."}]}] for p in batch]
    ehr_analyses = generate_response(prompts, model, processor, max_tokens)
    # CXR Agent
    prompts = []
    for p in batch:
        content = []
        if p['cxr_image_path'] != 'CXR image not yet available (no valid image in first 48 hours).':
             if os.path.exists(p['cxr_image_path']):
                 try: content.append({"type": "image", "content": Image.open(p['cxr_image_path']).convert("RGB")})
                 except: pass
        content.append({"type": "text", "text": "Analyze this CXR for findings indicating high mortality risk. Provide a concise one-sentence summary."})
        prompts.append([{"role": "user", "content": content}])
    cxr_summaries = generate_response(prompts, model, processor, max_tokens)
    # Notes Agent
    prompts = [[{"role": "user", "content": [{"type": "text", "text": f"Summarize the patient's condition for prognosis.\n\nPatient Summary:\n{p['patient_summary_text']}\n\nRadiology Report:\n{p['radiology_report_text']}\n\nProvide a concise one-sentence summary."}]}] for p in batch]
    notes_summaries = generate_response(prompts, model, processor, max_tokens)
    # --- End Specialist Analyses ---

    # Coordinator Agent
    coordinator_prompts = []
    for i in range(len(batch)):
        # MODIFIED: Changed prompt to ask for probability
        prompt = (
            "You are the lead physician. Based *only* on the following specialist reports, "
            "what is the in-hospital mortality probability? "
            "Respond *only* with the probability as a decimal (e.g., '0.35').\n\n"
            "--- DATA ---\n"
            f"Radiology:\n{cxr_summaries[i]}\n\n"
            f"Clinical Notes:\n{notes_summaries[i]}\n\n"
            f"EHR Data Analysis:\n{ehr_analyses[i]}\n\n"
            "--- PROBABILITY ---"
        )
        coordinator_prompts.append([{"role": "user", "content": [{"type": "text", "text": prompt}]}])

    # MODIFIED: Call new function
    texts, exp_probs, imp_probs = generate_probability_with_scores(
        coordinator_prompts, model, processor, max_tokens=10
    )
    modality_reqs = [{'patient_summary': 1, 'ehr_timeseries': 1, 'cxr': 1, 'radiology_report': 1}] * len(batch)
    return _format_batch_results(batch, texts, exp_probs, imp_probs, modality_reqs)

def _run_majority_vote_batch(batch, model, processor, args):
    """MODIFIED: Implements the "MajorityVote" architecture (now "AverageProbability")."""
    
    # --- 1. Get individual probabilities from each specialist agent ---
    
    # EHR Agent's Vote
    prompts_ehr = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are an EHR specialist. Based *only* on this EHR data, what is the mortality probability?
        Respond *only* with the probability as a decimal (e.g., '0.35').
        
        --- DATA ---
        {format_ehr_as_text(p['ehr_timeseries_path'])}
        
        --- PROBABILITY ---"""
                    }
                ]
            }
            ]
            for p in batch
        
    ]

    texts_ehr, exp_ehr, imp_ehr = generate_probability_with_scores(prompts_ehr, model, processor, max_tokens=10)

    # CXR Agent's Vote
    prompts_cxr = []
    for p in batch:
        content = []
        if p['cxr_image_path'] != 'CXR image not yet available (no valid image in first 48 hours).':
             if os.path.exists(p['cxr_image_path']):
                 try: content.append({"type": "image", "content": Image.open(p['cxr_image_path']).convert("RGB")})
                 except: pass
        content.append({"type": "text", "text": "Based *only* on this CXR, what is the mortality probability? Respond *only* with the probability as a decimal (e.g., '0.35')."})
        prompts_cxr.append([{"role": "user", "content": content}])
    texts_cxr, exp_cxr, imp_cxr = generate_probability_with_scores(prompts_cxr, model, processor, max_tokens=10)

    # Notes Agent's Vote
    prompts_notes = [[{"role": "user", "content": [{"type": "text", "text": f"Based *only* on these notes, what is the mortality probability? Respond *only* with the probability as a decimal (e.g., '0.35').\n\nSummary:\n{p['patient_summary_text']}\n\nRadiology Report:\n{p['radiology_report_text']}"}]}] for p in batch]
    texts_notes, exp_notes, imp_notes = generate_probability_with_scores(prompts_notes, model, processor, max_tokens=10)

    # --- 2. Average the probabilities ---
    batch_final_texts = []
    batch_final_exp_probs = []
    batch_final_imp_probs = []
    
    for i in range(len(batch)):
        # Average the explicit probabilities
        final_exp_prob = np.mean([exp_ehr[i], exp_cxr[i], exp_notes[i]])
        batch_final_exp_probs.append(final_exp_prob)
        
        # Average the implicit probabilities
        final_imp_prob = np.mean([imp_ehr[i], imp_cxr[i], imp_notes[i]])
        batch_final_imp_probs.append(final_imp_prob)
        
        # Combine text for logging
        final_text = f"EHR: {texts_ehr[i]} | CXR: {texts_cxr[i]} | Notes: {texts_notes[i]}"
        batch_final_texts.append(final_text)

    modality_reqs = [{'patient_summary': 1, 'ehr_timeseries': 1, 'cxr': 1, 'radiology_report': 1}] * len(batch)
    # Pass the averaged probabilities to the formatter
    return _format_batch_results(batch, batch_final_texts, batch_final_exp_probs, batch_final_imp_probs, modality_reqs)

def _run_debate_batch(batch, model, processor, args):
    """MODIFIED: Implements the multi-round "Debate" architecture"""
    
    # --- ROUND 1 & 2: Initial Analysis and Rebuttal (Unchanged) ---
    print("--- Starting Debate: Round 1 ---")
    # ... (R1: prompts_ehr1, analyses_ehr1) ...
    # R1: EHR Agent (Batched)
    prompts_ehr1 = []
    for p in batch:
        prompt_text = f"**Round 1/3**: Analyze this EHR data and form an initial opinion on mortality. State your reasoning and conclude with 'Decision: Yes' or 'Decision: No'.\n\n{format_ehr_as_text(p['ehr_timeseries_path'])}"
        prompts_ehr1.append([{"role": "user", "content": [{"type": "text", "text": prompt_text}]}])
    analyses_ehr1 = generate_response(prompts_ehr1, model, processor, args.max_new_tokens)

    # R1: CXR Agent (Batched)
    prompts_cxr1 = []
    for p in batch:
        content_cxr1 = []
        if p['cxr_image_path'] != 'CXR image not yet available (no valid image in first 48 hours).':
             if os.path.exists(p['cxr_image_path']):
                 try: content_cxr1.append({"type": "image", "content": Image.open(p['cxr_image_path']).convert("RGB")})
                 except: pass
        content_cxr1.append({"type": "text", "text": "**Round 1/3**: Analyze this CXR and form an initial opinion on mortality. State your reasoning and conclude with 'Decision: Yes' or 'Decision: No'."})
        prompts_cxr1.append([{"role": "user", "content": content_cxr1}])
    analyses_cxr1 = generate_response(prompts_cxr1, model, processor, args.max_new_tokens)
    
    # R1: Notes Agent (Batched)
    prompts_notes1 = []
    for p in batch:
        prompt_text = f"**Round 1/3**: Analyze these notes and form an initial opinion on mortality. State your reasoning and conclude with 'Decision: Yes' or 'Decision: No'.\n\nSummary:\n{p['patient_summary_text']}\n\nRadiology Report:\n{p['radiology_report_text']}"
        prompts_notes1.append([{"role": "user", "content": [{"type": "text", "text": prompt_text}]}])
    analyses_notes1 = generate_response(prompts_notes1, model, processor, args.max_new_tokens)

    print("--- Starting Debate: Round 2 ---")
    # ... (R2: prompts_ehr2, analyses_ehr2, etc.) ...
    # R2: EHR Agent (Batched)
    prompts_ehr2 = []
    for i in range(len(batch)):
        text = f"**Round 2/3**: You are the EHR agent. Your initial analysis was:\n'{analyses_ehr1[i]}'\n\nThe other specialists found:\n- CXR Agent: '{analyses_cxr1[i]}'\n- Notes Agent: '{analyses_notes1[i]}'\n\nConsidering their findings, re-evaluate your position. Provide updated reasoning and conclude with 'Decision: Yes' or 'Decision: No'."
        prompts_ehr2.append([{"role": "user", "content": [{"type": "text", "text": text}]}])
    analyses_ehr2 = generate_response(prompts_ehr2, model, processor, args.max_new_tokens)

    # R2: CXR Agent (Batched)
    prompts_cxr2 = []
    for i in range(len(batch)):
        text = f"**Round 2/3**: You are the CXR agent. Your initial analysis was:\n'{analyses_cxr1[i]}'\n\nThe other specialists found:\n- EHR Agent: '{analyses_ehr1[i]}'\n- Notes Agent: '{analyses_notes1[i]}'\n\nConsidering their findings, re-evaluate your position. Provide updated reasoning and conclude with 'Decision: Yes' or 'Decision: No'."
        # Must re-send the image (content from prompts_cxr1)
        original_content = prompts_cxr1[i][0]['content'][:-1]
        new_content = original_content + [{"type": "text", "text": text}]
        prompts_cxr2.append([{"role": "user", "content": new_content}])
    analyses_cxr2 = generate_response(prompts_cxr2, model, processor, args.max_new_tokens)

    # R2: Notes Agent (Batched)
    prompts_notes2 = []
    for i in range(len(batch)):
        text = f"**Round 2/3**: You are the Notes agent. Your initial analysis was:\n'{analyses_notes1[i]}'\n\nThe other specialists found:\n- EHR Agent: '{analyses_ehr1[i]}'\n- CXR Agent: '{analyses_cxr1[i]}'\n\nConsidering their findings, re-evaluate your position. Provide updated reasoning and conclude with 'Decision: Yes' or 'Decision: No'."
        prompts_notes2.append([{"role": "user", "content": [{"type": "text", "text": text}]}])
    analyses_notes2 = generate_response(prompts_notes2, model, processor, args.max_new_tokens)

    # --- ROUND 3: Final Probabilities (MODIFIED) ---
    print("--- Starting Debate: Round 3 (Final Probability) ---")
    
    # R3: EHR Agent
    prompts_ehr3 = []
    for i in range(len(batch)):
        prompt_text = (
            "**Round 3/3 (Final)**: You are the EHR specialist. After seeing all arguments, what is the final mortality probability?"
            " Respond *only* with the probability as a decimal (e.g., '0.35').\n\n"
            "--- DATA (Round 2 Arguments) ---\n"
            f"- EHR (You): '{analyses_ehr2[i]}'\n"
            f"- CXR: '{analyses_cxr2[i]}'\n"
            f"- Notes: '{analyses_notes2[i]}'\n\n"
            "--- PROBABILITY ---"
        )
        prompts_ehr3.append([{"role": "user", "content": [{"type": "text", "text": prompt_text}]}])
    texts_ehr, exp_ehr, imp_ehr = generate_probability_with_scores(prompts_ehr3, model, processor, max_tokens=10)
    
    # R3: CXR Agent
    prompts_cxr3 = []
    for i in range(len(batch)):
        prompt_text = (
            "**Round 3/3 (Final)**: You are the CXR specialist. After seeing all arguments, what is the final mortality probability?"
            " Respond *only* with the probability as a decimal (e.g., '0.35').\n\n"
            "--- DATA (Round 2 Arguments) ---\n"
            f"- EHR: '{analyses_ehr2[i]}'\n"
            f"- CXR (You): '{analyses_cxr2[i]}'\n"
            f"- Notes: '{analyses_notes2[i]}'\n\n"
            "--- PROBABILITY ---"
        )
        prompts_cxr3.append([{"role": "user", "content": [{"type": "text", "text": prompt_text}]}])
    texts_cxr, exp_cxr, imp_cxr = generate_probability_with_scores(prompts_cxr3, model, processor, max_tokens=10)

    # R3: Notes Agent
    prompts_notes3 = []
    for i in range(len(batch)):
        prompt_text = (
            "**Round 3/3 (Final)**: You are the Notes specialist. After seeing all arguments, what is the final mortality probability?"
            " Respond *only* with the probability as a decimal (e.g., '0.35').\n\n"
            "--- DATA (Round 2 Arguments) ---\n"
            f"- EHR: '{analyses_ehr2[i]}'\n"
            f"- CXR: '{analyses_cxr2[i]}'\n"
            f"- Notes (You): '{analyses_notes2[i]}'\n\n"
            "--- PROBABILITY ---"
        )
        prompts_notes3.append([{"role": "user", "content": [{"type": "text", "text": prompt_text}]}])
    texts_notes, exp_notes, imp_notes = generate_probability_with_scores(prompts_notes3, model, processor, max_tokens=10)
    
    # --- Average Final Probabilities (MODIFIED) ---
    batch_final_texts = []
    batch_final_exp_probs = []
    batch_final_imp_probs = []

    for i in range(len(batch)):
        final_exp_prob = np.mean([exp_ehr[i], exp_cxr[i], exp_notes[i]])
        batch_final_exp_probs.append(final_exp_prob)
        
        final_imp_prob = np.mean([imp_ehr[i], imp_cxr[i], imp_notes[i]])
        batch_final_imp_probs.append(final_imp_prob)

        final_text = f"EHR: {texts_ehr[i]} | CXR: {texts_cxr[i]} | Notes: {texts_notes[i]}"
        batch_final_texts.append(final_text)

    modality_reqs = [{'patient_summary': 1, 'ehr_timeseries': 1, 'cxr': 1, 'radiology_report': 1}] * len(batch)
    return _format_batch_results(batch, batch_final_texts, batch_final_exp_probs, batch_final_imp_probs, modality_reqs)

def _run_dual_agent_batch(batch, model, processor, args):
    """MODIFIED: Implements the 'DualAgent' (Selector -> Decider) approach."""
    
    # --- AGENT 1: SELECTOR AGENT (Unchanged) ---
    print("--- Starting DualAgent: Agent 1 (Selector) ---")
    selector_prompts = []
    available_modalities = "EHR_TIMESERIES, RADIOLOGY_REPORT, CXR_IMAGE"
    
    for p in batch:
        prompt_text = (
            "You are a clinical triage agent. Your task is to determine which *additional* data modalities are "
            "necessary to predict in-hospital mortality, based *only* on the following patient summary. "
            "The summary is *always* provided to the final decision agent.\n\n"
            f"Patient Summary:\n{p['patient_summary_text']}\n\n"
            f"Which of the following *additional* modalities do you require? "
            f"Available: [{available_modalities}]\n"
            "Respond with a comma-separated list of the modalities you need (e.g., 'EHR_TIMESERIES, CXR_IMAGE'). "
            "If the summary is sufficient, respond with 'NONE'."
        )
        selector_prompts.append([{"role": "user", "content": [{"type": "text", "text": prompt_text}]}])
    
    selector_responses = generate_response(selector_prompts, model, processor, max_tokens=20)
    
    # --- Parse Selections & Build Prompts for AGENT 2 (Unchanged) ---
    print("--- Starting DualAgent: Agent 2 (Decision) ---")
    decision_prompts = []
    modality_requests_list = [] 
    
    ehr_texts = [format_ehr_as_text(p['ehr_timeseries_path']) for p in batch]
    
    for i, patient in enumerate(batch):
        selection_text = selector_responses[i].upper()
        
        req_ehr = "EHR_TIMESERIES" in selection_text
        req_report = "RADIOLOGY_REPORT" in selection_text
        req_cxr = "CXR_IMAGE" in selection_text
        
        modality_requests = {
            'patient_summary': 1, 
            'ehr_timeseries': 1 if req_ehr else 0,
            'radiology_report': 1 if req_report else 0,
            'cxr': 0 # Will be set to 1 if requested AND available
        }
        
        content = []
        # Build the data block first
        data_builder = [f"Patient Summary:\n{patient['patient_summary_text']}\n"]
        
        if req_cxr and patient['cxr_image_path'] != 'CXR image not yet available (no valid image in first 48 hours).':
            if os.path.exists(patient['cxr_image_path']):
                try:
                    content.append({"type": "image", "content": Image.open(patient['cxr_image_path']).convert("RGB")})
                    modality_requests['cxr'] = 1
                    data_builder.append("CXR Image: [Image is attached]\n")
                except Exception as e:
                    print(f"Warning: Could not load image {patient['cxr_image_path']}: {e}")
                    data_builder.append("CXR Image: [Error loading image, was requested but unavailable]\n")
        elif req_cxr:
            data_builder.append("CXR Image: [Requested by triage, but not available in patient record]\n")

        if req_report:
            data_builder.append(f"Radiology Report:\n{patient['radiology_report_text']}\n")
        
        if req_ehr:
            data_builder.append(f"{ehr_texts[i]}\n")
        
        # MODIFIED: Changed prompt to Instruction-First, Data-Second, Primer-Last
        prompt_text = (
            "You are a final decision agent. A triage agent reviewed this case and requested the data below. "
            "Based *only* on this provided data, "
            "what is the patient's probability of in-hospital mortality? "
            "Respond *only* with the probability as a decimal (e.g., '0.35').\n\n"
            "--- DATA ---\n"
            f"{''.join(data_builder)}\n"
            "--- PROBABILITY ---"
        )
        # ================== END FIX ==================
        
        content.append({"type": "text", "text": prompt_text})
        decision_prompts.append([{"role": "user", "content": content}])
        modality_requests_list.append(modality_requests)
    
    # --- AGENT 2: DECISION AGENT (MODIFIED) ---
    texts, exp_probs, imp_probs = generate_probability_with_scores(
        decision_prompts, model, processor, max_tokens=10
    )
    
    return _format_batch_results(batch, texts, exp_probs, imp_probs, modality_requests_list)


# --------------------------------------------------------------------
# 🚦 4. Main Dispatcher Function 
# --------------------------------------------------------------------

def initialize_agent_setup(batch, args):
    """MODIFIED: Main entry point. Loads model and selects agent."""
    # MODIFIED: No longer need yes_id, no_id
    model, processor = get_model_and_processor(args)
    agent_setup_name = args.agent_setup
    
    if agent_setup_name == 'SingleAgent':
        return _run_single_agent_batch(batch, model, processor, args)
    
    elif agent_setup_name in ['MultiAgent', 'MedCollab']:
        return _run_multi_agent_batch(batch, model, processor, args)

    elif agent_setup_name == 'MajorityVote':
        return _run_majority_vote_batch(batch, model, processor, args)

    elif agent_setup_name == 'Debate':
        return _run_debate_batch(batch, model, processor, args)
        
    elif agent_setup_name == 'DualAgent':
        return _run_dual_agent_batch(batch, model, processor, args)
        
    else:
        raise ValueError(f"Unknown agent_setup: '{agent_setup_name}'. Please choose a valid name.")