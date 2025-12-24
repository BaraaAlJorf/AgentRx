import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM
from PIL import Image
from datasets.data_utils import load_few_shot_data

# --------------------------------------------------------------------
# 1. Model & Processor Setup
# --------------------------------------------------------------------
MODEL_CACHE = {}
SUPPORT_SET_CACHE = None 
_GLOBAL_DEBUG_PRINT_COUNT = 0  # Restored global counter

def get_model_and_processor(args):
    """Loads and caches the model/processor with strict class routing."""
    model_id = args.model_id
    if model_id in MODEL_CACHE:
        return MODEL_CACHE[model_id]

    print(f"Initializing Model and Processor for '{model_id}'...")
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

    load_kwargs = {
        "token": hf_token,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }

    try:
        # --- 1. Load Processor ---
        if "Qwen" in model_id:
            processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
            # CRITICAL FIX for Qwen batching
            if hasattr(processor, 'tokenizer'):
                print("⚡ Applying Left-Padding fix for Qwen...")
                processor.tokenizer.padding_side = "left"
                processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        else:
            processor = AutoProcessor.from_pretrained(model_id, token=hf_token, trust_remote_code=True)

        # --- 2. Load Model ---
        model = None
        if "Qwen" in model_id:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                print(f"Loading {model_id} as Qwen2_5_VLForConditionalGeneration...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
            except ImportError:
                print(f"Qwen2_5_VL class not found, falling back to AutoModelForCausalLM...")
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        elif "InternVL" in model_id:
            print(f"Loading {model_id} as AutoModel (Remote Code)...")
            model = AutoModel.from_pretrained(model_id, **load_kwargs)
        elif "Phi-4" in model_id:
            print(f"Loading {model_id} as AutoModelForCausalLM...")
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        else:
            print(f"Loading {model_id} with generic AutoModel fallback...")
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            except:
                model = AutoModel.from_pretrained(model_id, **load_kwargs)

        model.eval()
        
        # --- 3. Tokenizer Handling ---
        try:
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            candidates_yes = ["Yes", " Yes", "yes", " yes"]
            candidates_no = ["No", " No", "no", " no"]
            yes_id, no_id = None, None
            
            for c in candidates_yes:
                ids = tokenizer.encode(c, add_special_tokens=False)
                if len(ids) == 1: 
                    yes_id = ids[0]
                    break
            for c in candidates_no:
                ids = tokenizer.encode(c, add_special_tokens=False)
                if len(ids) == 1: 
                    no_id = ids[0]
                    break
            
            if yes_id is None: yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
            if no_id is None: no_id = tokenizer.encode("No", add_special_tokens=False)[0]

            print(f"✅ Using Token IDs - Yes: {yes_id}, No: {no_id}")
            processor.yes_token_id = yes_id
            processor.no_token_id = no_id

        except Exception as e:
            print(f"Warning: Could not set canonical Yes/No IDs: {e}")
            processor.yes_token_id = None
            processor.no_token_id = None

        print(f"✅ Model loaded successfully.")
        MODEL_CACHE[model_id] = (model, processor)
        return model, processor

    except Exception as e:
        print(f"❌ Failed to load model '{model_id}'. Error: {e}")
        exit()

# --------------------------------------------------------------------
# 2. Inference Helpers & Debugging
# --------------------------------------------------------------------

def _print_debug_sample(args, batch, prompts, tag="DEBUG SAMPLE"):
    """Prints prompt content WITHOUT truncating."""
    global _GLOBAL_DEBUG_PRINT_COUNT
    debug_limit = getattr(args, 'debug_samples', 0)
    
    if _GLOBAL_DEBUG_PRINT_COUNT < debug_limit and len(batch) > 0:
        print(f"\n--- {tag} ({_GLOBAL_DEBUG_PRINT_COUNT + 1}/{debug_limit}) ---")
        p = batch[0]
        print(f"Stay ID: {p.get('stay_id', 'N/A')}")
        try:
            content_list = prompts[0][0]['content']
            for item in content_list:
                if item['type'] == 'text':
                    # --- UPDATED: No slicing/truncation here ---
                    print(f"[TEXT BLOCK]:\n{item['text']}")
                    print("-" * 20) 
                elif item['type'] == 'image':
                    img_info = item['image']
                    print(f"[IMAGE BLOCK]: {img_info} (Size: {img_info.size})")
        except: pass
        print("="*60 + "\n")
        _GLOBAL_DEBUG_PRINT_COUNT += 1
        
def _prepare_inputs_for_vlm(prompts, processor, device):
    is_qwen = hasattr(processor, "image_processor") and "Qwen" in processor.image_processor.__class__.__name__

    if is_qwen:
        try:
            from qwen_vl_utils import process_vision_info
            texts = [
                processor.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                for p in prompts
            ]
            image_inputs, video_inputs = process_vision_info(prompts)
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            return inputs.to(device)
        except ImportError:
            print("[Warning] 'qwen_vl_utils' not found. Falling back.")
            pass

    inputs = processor.apply_chat_template(
        prompts, 
        add_generation_prompt=True, 
        tokenize=True,
        return_tensors="pt", 
        padding=True, 
        return_dict=True
    ).to(device)
    return inputs

def generate_response(prompts, model, processor, max_tokens, **generation_kwargs):
    inputs = _prepare_inputs_for_vlm(prompts, processor, model.device)
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                use_cache=True,
                **generation_kwargs # Allow sampling args (do_sample, temperature, etc.)
            )
    except RuntimeError as e:
        if "out of memory" in str(e).lower(): return ["Error: OOM"] * len(prompts)
        raise
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)

def generate_yes_no_probability(prompts, model, processor, max_tokens=1):
    inputs = _prepare_inputs_for_vlm(prompts, processor, model.device)
    try:
        with torch.inference_mode():
            outputs = model(
                **inputs,
                # max_new_tokens=max_tokens, use_cache=True,
                # return_dict_in_generate=True, output_scores=True
            )
            # Use direct forward pass logits for last token
            next_token_logits = outputs.logits[:, -1, :]
    except RuntimeError as e:
        if "out of memory" in str(e).lower(): return ["N/A"] * len(prompts), np.full(len(prompts), 0.5)
        raise

    yes_probs = np.zeros(len(prompts), dtype=np.float32)
    if hasattr(processor, 'yes_token_id') and processor.yes_token_id is not None:
        try:
            yes_score = next_token_logits[:, processor.yes_token_id]
            no_score = next_token_logits[:, processor.no_token_id]
            yes_no_logits = torch.stack([no_score, yes_score], dim=1)
            probs = F.softmax(yes_no_logits, dim=-1)
            yes_probs = probs[:, 1].cpu().float().numpy()
            if np.isnan(yes_probs).any():
                yes_probs = np.nan_to_num(yes_probs, nan=0.0)
        except Exception as e:
            pass

    # Decode top token for sanity check (though not strictly needed for probs)
    top_tokens = torch.argmax(next_token_logits, dim=-1)
    try:
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        decoded_texts = tokenizer.batch_decode(top_tokens.unsqueeze(-1), skip_special_tokens=True)
    except:
        decoded_texts = ["N/A"] * len(prompts)
        
    return [t.strip() for t in decoded_texts], yes_probs

def _format_batch_results(batch, gen_texts, explicit_probs, modality_requests_list):
    batch_results = []
    for i, patient_data in enumerate(batch):
        exp_prob = float(explicit_probs[i])
        result = {
            'subject_id': patient_data.get('subject_id'),
            'stay_id': patient_data['stay_id'], 
            'ground_truth': patient_data['labels'],
            'predictions': {
                'in_hospital_mortality_48hr': 1 if exp_prob > 0.5 else 0, 
                'mortality_probability': exp_prob,
                'mortality_probability_text': gen_texts[i]
            },
            'modality_requests': modality_requests_list[i],
            'modality_availability': {
                'cxr': patient_data.get('cxr_image_path', '') != 'CXR not available',
                'radiology_report': patient_data.get('radiology_report_text', '') != 'Radiology report not available'
            }
        }
        batch_results.append(result)
    return batch_results

# --------------------------------------------------------------------
# 3. Agent Architecture Helpers
# --------------------------------------------------------------------

def _parse_allowed_modalities(args):
    if not hasattr(args, 'modalities') or not args.modalities:
        return ['ehr', 'cxr', 'rr', 'ps']
    return [m.strip().lower() for m in args.modalities.split('-')]

def _build_prompt_content(patient_data, ehr_text, allowed_modalities, is_training_example=False, outcome=None, prompt_type="standard", previous_reasoning=None, feedback = None):
    content = []
    has_image = False
    
    if 'cxr' in allowed_modalities:
        if 'pil_image' in patient_data and patient_data['pil_image'] is not None:
             content.append({"type": "image", "image": patient_data['pil_image']})
             has_image = True
        elif patient_data.get('cxr_image_path') != 'CXR not available':
            if os.path.exists(patient_data['cxr_image_path']):
                try:
                    content.append({"type": "image", "image": Image.open(patient_data['cxr_image_path']).convert("RGB")})
                    has_image = True
                except: pass
    
    data_parts = ["--- Patient DATA ---"]
    if 'ps' in allowed_modalities:
        data_parts.append(f"Patient Summary:\n{patient_data.get('patient_summary_text', '')}")
    if 'rr' in allowed_modalities:
        data_parts.append(f"Radiology Reports:\n{patient_data.get('radiology_report_text', '')}")
    if 'cxr' in allowed_modalities:
        data_parts.append(f"Chest X-ray: [{'Attached Above' if has_image else 'Not Available'}]")
    if 'ehr' in allowed_modalities:
        data_parts.append(f"Electronic Health Records:\n{ehr_text}")

    data_block = "\n\n".join(data_parts)

    # Prompt Logic
    if outcome is not None:
        # Few-Shot Example
        outcome_str = "Yes" if outcome == 1 else "No"
        prompt_text = (
            f"{data_block}\n\n"
            "--- DECISION ---\n"
            "Does this patient die in the ICU? Answer only using one word - Yes or No?\n"
            f"Answer: {outcome_str}\n\n"
            "--------------------------------------------------\n"
        )
    elif prompt_type == "cot_reasoning":
        # CoT Step 1
        prompt_text = (
            f"{data_block}\n\n"
            "--- ANALYSIS ---\n"
            "Analyze the patient's condition step by step. Consider vitals, labs, and history. "
            "Identify key risk factors for imminent mortality.\n"
            "Reasoning:"
        )
        
    elif prompt_type == "refine_feedback":
        # Self-Refine Step 1: Ask for Critique
        prompt_text = (
            f"{data_block}\n\n"
            "--- PREVIOUS ANALYSIS ---\n"
            f"{previous_reasoning}\n\n"
            "--- TASK ---\n"
            "Review the analysis above. Identify any missing vital signs, logical gaps, or overlooked risk factors in the data. "
            "Provide constructive feedback to improve the mortality assessment. "
            "Do NOT output a final decision yet, just the feedback.\n"
            "Feedback:"
        )
        
    elif prompt_type == "refine_update":
        # Self-Refine Step 2: Ask for Rewrite
        prompt_text = (
            f"{data_block}\n\n"
            "--- PREVIOUS ANALYSIS ---\n"
            f"{previous_reasoning}\n\n"
            "--- FEEDBACK ---\n"
            f"{feedback}\n\n"
            "--- TASK ---\n"
            "Rewrite and improve the analysis based on the feedback. Be concise and clinical.\n"
            "Refined Analysis:"
        )    
        
    elif prompt_type == "cot_answer":
        # CoT Step 2 (Feed reasoning back)
        prompt_text = (
            f"{data_block}\n\n"
            "--- ANALYSIS ---\n"
            f"Model Reasoning: {previous_reasoning}\n\n"
            "--- DECISION ---\n"
            "Based on the analysis above, does this patient die in the ICU? Answer only using one word - Yes or No\n"
            "Answer: "
        )
    else:
        # Standard / Single Agent
        prompt_text = (
            f"{data_block}\n\n"
            "--- DECISION ---\n"
            "Does this patient die in the ICU? Answer only using one word - Yes or No\n"
            "Answer: " 
        )

    content.append({"type": "text", "text": prompt_text})
    return content, has_image

# --------------------------------------------------------------------
# 4. Agent Implementations
# --------------------------------------------------------------------

def _run_single_agent_batch(batch, model, processor, args):
    prompts = []
    modality_requests_list = []
    allowed_mods = _parse_allowed_modalities(args)
    
    for patient in batch:
        ehr_text = patient.get('ehr_text', "EHR Data Not Available")
        content, has_image = _build_prompt_content(
            patient, ehr_text, allowed_mods, is_training_example=False, prompt_type="standard"
        )
        system_msg = {"type": "text", "text": "You are an expert ICU risk prediction model. This patient was just admitted.\n\n"}
        full_content = [system_msg] + content
        prompts.append([{"role": "user", "content": full_content}])
        
        modality_requests_list.append({
            'patient_summary': 1 if 'ps' in allowed_mods else 0,
            'ehr_timeseries': 1 if 'ehr' in allowed_mods else 0,
            'radiology_report': 1 if 'rr' in allowed_mods else 0,
            'cxr': 1 if 'cxr' in allowed_mods and has_image else 0
        })

    # --- DEBUGGING HOOK ---
    _print_debug_sample(args, batch, prompts, tag="Standard Agent")
    
    texts, exp_probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
    return _format_batch_results(batch, texts, exp_probs, modality_requests_list)

def _run_few_shot_batch(batch, model, processor, args):
    global SUPPORT_SET_CACHE
    n_shots = args.num_shots if args.num_shots is not None else 2
    
    if SUPPORT_SET_CACHE is None:
        if not hasattr(args, 'few_shot_data_path') or not args.few_shot_data_path:
             print("[Error] 'FewShot' agent requires --few_shot_data_path argument.")
             return []
        print(f"Loading few-shot support set from {args.few_shot_data_path}...")
        SUPPORT_SET_CACHE = load_few_shot_data(args.few_shot_data_path, num_shots=n_shots)

    allowed_mods = _parse_allowed_modalities(args)
    prompts = []
    modality_requests_list = []

    for patient in batch:
        full_content = []
        full_content.append({"type": "text", "text": "You are an expert ICU risk prediction model. Here are some examples of patient data and their outcomes.\n\n"})
        
        for shot_patient in SUPPORT_SET_CACHE:
            shot_outcome = shot_patient['labels']['in_hospital_mortality_48hr']
            shot_content, _ = _build_prompt_content(
                shot_patient, shot_patient.get('ehr_text', ''), allowed_mods, 
                is_training_example=True, outcome=shot_outcome
            )
            full_content.extend(shot_content)
            
        full_content.append({"type": "text", "text": "Now, analyze this new patient:\n"})
        target_content, has_image = _build_prompt_content(
            patient, patient.get('ehr_text', ''), allowed_mods, is_training_example=False, prompt_type="standard"
        )
        full_content.extend(target_content)
        prompts.append([{"role": "user", "content": full_content}])
        
        modality_requests_list.append({
            'patient_summary': 1 if 'ps' in allowed_mods else 0,
            'ehr_timeseries': 1 if 'ehr' in allowed_mods else 0,
            'radiology_report': 1 if 'rr' in allowed_mods else 0,
            'cxr': 1 if 'cxr' in allowed_mods and has_image else 0
        })

    # --- DEBUGGING HOOK ---
    _print_debug_sample(args, batch, prompts, tag="Few-Shot")

    texts, exp_probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
    return _format_batch_results(batch, texts, exp_probs, modality_requests_list)
    
def _run_single_agent_cot_batch(batch, model, processor, args):
    allowed_mods = _parse_allowed_modalities(args)
    modality_requests_list = []
    
    # 1. Reasoning
    reasoning_prompts = []
    for patient in batch:
        ehr_text = patient.get('ehr_text', "EHR Data Not Available")
        content, has_image = _build_prompt_content(
            patient, ehr_text, allowed_mods, prompt_type="cot_reasoning"
        )
        reasoning_prompts.append([{"role": "user", "content": content}])
        modality_requests_list.append({
            'patient_summary': 1 if 'ps' in allowed_mods else 0,
            'ehr_timeseries': 1 if 'ehr' in allowed_mods else 0,
            'radiology_report': 1 if 'rr' in allowed_mods else 0,
            'cxr': 1 if 'cxr' in allowed_mods and has_image else 0
        })

    _print_debug_sample(args, batch, reasoning_prompts, tag="CoT Step 1")
    # Standard generation (Greedy is default if no kwargs passed)
    reasoning_outputs = generate_response(reasoning_prompts, model, processor, max_tokens=256)
    
    # 2. Answer
    answer_prompts = []
    for i, patient in enumerate(batch):
        ehr_text = patient.get('ehr_text', "EHR Data Not Available")
        content, _ = _build_prompt_content(
            patient, ehr_text, allowed_mods, 
            prompt_type="cot_answer", 
            previous_reasoning=reasoning_outputs[i]
        )
        answer_prompts.append([{"role": "user", "content": content}])

    _print_debug_sample(args, batch, answer_prompts, tag="CoT Step 2")
    texts, exp_probs = generate_yes_no_probability(answer_prompts, model, processor, max_tokens=1)
    return _format_batch_results(batch, texts, exp_probs, modality_requests_list)

def _run_single_agent_self_consistency_batch(batch, model, processor, args):
    """
    Self-Consistency:
    1. Sample multiple diverse CoT reasoning paths (do_sample=True, temp=0.7).
    2. Aggregate the final answers (Average Probabilities).
    """
    n_samples = getattr(args, 'consistency_samples', 5)
    print(f"   [Self-Consistency] Sampling {n_samples} CoT paths per patient...")
    
    allowed_mods = _parse_allowed_modalities(args)
    
    # Accumulators
    sum_probs = np.zeros(len(batch), dtype=np.float32)
    last_reasoning_text = [""] * len(batch) # Just for logging
    modality_requests_list = []
    for k in range(n_samples):
        # Step 1: Generate DIVERSE reasoning
        reasoning_prompts = []
        for patient in batch:
            ehr_text = patient.get('ehr_text', "EHR Data Not Available")
            content, has_image = _build_prompt_content(patient, ehr_text, allowed_mods, prompt_type="cot_reasoning")
            reasoning_prompts.append([{"role": "user", "content": content}])
            modality_requests_list.append({
            'patient_summary': 1 if 'ps' in allowed_mods else 0,
            'ehr_timeseries': 1 if 'ehr' in allowed_mods else 0,
            'radiology_report': 1 if 'rr' in allowed_mods else 0,
            'cxr': 1 if 'cxr' in allowed_mods and has_image else 0
            })
            
        # Critical: Use Sampling kwargs
        reasoning_outputs = generate_response(
            reasoning_prompts, model, processor, max_tokens=256, 
            do_sample=True, temperature=0.7, top_k=50
        )
        
        # Step 2: Get Probability for this specific path
        answer_prompts = []
        for i, patient in enumerate(batch):
            ehr_text = patient.get('ehr_text', "EHR Data Not Available")
            content, _ = _build_prompt_content(
                patient, ehr_text, allowed_mods, prompt_type="cot_answer", previous_reasoning=reasoning_outputs[i]
            )
            answer_prompts.append([{"role": "user", "content": content}])
            
        _, exp_probs = generate_yes_no_probability(answer_prompts, model, processor, max_tokens=1)
        
        sum_probs += exp_probs
        if k == 0: 
            _print_debug_sample(args, batch, reasoning_prompts, tag="SC Sample 1 Reasoning")
            last_reasoning_text = reasoning_outputs 

    avg_probs = sum_probs / n_samples
    final_texts = [f"[SC-{n_samples}] {txt}..." for txt in last_reasoning_text]
    
    return _format_batch_results(batch, final_texts, avg_probs, modality_requests_list)
    
def _run_self_refine_batch(batch, model, processor, args):
    """
    Self-Refine: Iterative Refinement (arXiv:2303.17651).
    Flow: Generate -> Feedback -> Refine -> Answer
    """
    allowed_mods = _parse_allowed_modalities(args)
    iterations = getattr(args, 'refine_iterations', 1) # Default 1 refinement step
    modality_requests_list = []
    
    # 1. Initial Generation
    prompts = []
    for patient in batch:
        ehr_text = patient.get('ehr_text', "")
        content, _ = _build_prompt_content(patient, ehr_text, allowed_mods, prompt_type="cot_reasoning")
        prompts.append([{"role": "user", "content": content}])
        modality_requests_list.append({
            'patient_summary': 1 if 'ps' in allowed_mods else 0,
            'ehr_timeseries': 1 if 'ehr' in allowed_mods else 0,
            'radiology_report': 1 if 'rr' in allowed_mods else 0,
            'cxr': 1 if 'cxr' in allowed_mods and has_image else 0
            })
    
    _print_debug_sample(args, batch, prompts, tag="Refine Step 0 (Initial)")
    current_outputs = generate_response(prompts, model, processor, max_tokens=256)
    
    # 2. Refinement Loop
    for k in range(iterations):
        # A. Feedback Step
        feedback_prompts = []
        for i, patient in enumerate(batch):
            ehr_text = patient.get('ehr_text', "")
            content, _ = _build_prompt_content(
                patient, ehr_text, allowed_mods, 
                prompt_type="refine_feedback", 
                previous_reasoning=current_outputs[i]
            )
            feedback_prompts.append([{"role": "user", "content": content}])
            
        feedbacks = generate_response(feedback_prompts, model, processor, max_tokens=128)
        _print_debug_sample(args, batch, feedback_prompts, tag=f"Refine Step {k+1}A (Feedback)")

        # B. Update Step
        update_prompts = []
        for i, patient in enumerate(batch):
            ehr_text = patient.get('ehr_text', "")
            content, _ = _build_prompt_content(
                patient, ehr_text, allowed_mods, 
                prompt_type="refine_update", 
                previous_reasoning=current_outputs[i],
                feedback=feedbacks[i]
            )
            update_prompts.append([{"role": "user", "content": content}])
            
        current_outputs = generate_response(update_prompts, model, processor, max_tokens=256)
        _print_debug_sample(args, batch, update_prompts, tag=f"Refine Step {k+1}B (Update)")

    # 3. Final Prediction using Refined Reasoning
    answer_prompts = []
    for i, patient in enumerate(batch):
        ehr_text = patient.get('ehr_text', "")
        content, _ = _build_prompt_content(
            patient, ehr_text, allowed_mods, 
            prompt_type="cot_answer", 
            previous_reasoning=current_outputs[i]
        )
        answer_prompts.append([{"role": "user", "content": content}])
        
    texts, exp_probs = generate_yes_no_probability(answer_prompts, model, processor, max_tokens=1)
    return _format_batch_results(batch, texts, exp_probs, modality_requests_list)

def _run_multi_agent_batch(batch, model, processor, args):
    max_tokens = args.max_new_tokens
    
    # Specialist 1: EHR
    prompts_ehr = [[{"role": "user", "content": [{"type": "text", "text": f"You are a clinical data analyst. Based on the following data, what is the patient's stability trend?\n\n{p.get('ehr_text', '')}\n\nProvide a concise one-sentence analysis."}]}] for p in batch]
    ehr_analyses = generate_response(prompts_ehr, model, processor, max_tokens)
    
    # Specialist 2: CXR
    prompts_cxr = []
    for p in batch:
        content = []
        if 'pil_image' in p and p['pil_image'] is not None:
             content.append({"type": "image", "image": p['pil_image']})
        elif p.get('cxr_image_path') != 'CXR not available' and os.path.exists(p['cxr_image_path']):
            try: content.append({"type": "image", "image": Image.open(p['cxr_image_path']).convert("RGB")})
            except: pass
        content.append({"type": "text", "text": "Analyze this CXR for findings indicating high mortality risk. Provide a concise one-sentence summary."})
        prompts_cxr.append([{"role": "user", "content": content}])
    cxr_summaries = generate_response(prompts_cxr, model, processor, max_tokens)
    
    # Specialist 3: Notes
    prompts_notes = [[{"role": "user", "content": [{"type": "text", "text": f"Summarize the patient's condition for prognosis.\n\nPatient Summary:\n{p['patient_summary_text']}\n\nRadiology Report:\n{p['radiology_report_text']}\n\nProvide a concise one-sentence summary."}]}] for p in batch]
    notes_summaries = generate_response(prompts_notes, model, processor, max_tokens)

    # Coordinator
    coordinator_prompts = []
    for i in range(len(batch)):
        prompt = (
            "You are the lead physician. Based *only* on the following specialist reports, "
            "what is the in-hospital mortality risk? "
            "Respond *only* with 'Yes' or 'No'.\n\n"
            "--- DATA ---\n"
            f"Radiology:\n{cxr_summaries[i]}\n\n"
            f"Clinical Notes:\n{notes_summaries[i]}\n\n"
            f"EHR Data Analysis:\n{ehr_analyses[i]}\n\n"
            "--- DECISION ---"
        )
        coordinator_prompts.append([{"role": "user", "content": [{"type": "text", "text": prompt}]}])

    # --- DEBUGGING HOOK (Coordinator only) ---
    _print_debug_sample(args, batch, coordinator_prompts)

    texts, exp_probs = generate_yes_no_probability(coordinator_prompts, model, processor, max_tokens=1)
    modality_reqs = [{'patient_summary': 1, 'ehr_timeseries': 1, 'cxr': 1, 'radiology_report': 1}] * len(batch)
    return _format_batch_results(batch, texts, exp_probs, modality_reqs)

def _run_majority_vote_uni_modal_batch(batch, model, processor, args):
    allowed_mods = _parse_allowed_modalities(args)
    batch_size = len(batch)
    all_voter_probs = []
    active_voters = []

    configs = [('ps', 'PS', "Predict mortality based ONLY on the summary."), 
               ('ehr', 'EHR', "Predict mortality based ONLY on the vitals."), 
               ('rr', 'RR', "Predict mortality based ONLY on the radiology report."), 
               ('cxr', 'CXR', "Predict mortality based ONLY on the X-ray.")]
    
    for mod_key, mod_name, sys_desc in configs:
        if mod_key in allowed_mods:
            prompts = []
            for p in batch:
                content, _ = _build_prompt_content(p, p.get('ehr_text', ""), [mod_key], prompt_type="standard")
                prompts.append([{"role": "user", "content": [{"type": "text", "text": f"{sys_desc}\n\n"}] + content}])
            _, probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
            all_voter_probs.append(probs)
            active_voters.append(mod_name)

    if not all_voter_probs: return _format_batch_results(batch, ["No Data"]*batch_size, np.zeros(batch_size), [])
    avg_probs = np.mean(all_voter_probs, axis=0)
    final_texts = [f"Vote UniModal ({'+'.join(active_voters)})"] * batch_size
    return _format_batch_results(batch, final_texts, avg_probs, [{'patient_summary': 1, 'ehr_timeseries': 1, 'radiology_report': 1, 'cxr': 1}] * batch_size)

# --------------------------------------------------------------------
# 6. Debate Implementations (Consensus Loop)
# --------------------------------------------------------------------

def _run_debate_unimodal_batch(batch, model, processor, args):
    """
    Consensus Debate (UniModal):
    Dynamically spawns agents for each available modality (PS, RR, CXR, EHR).
    They generate arguments, see peers', update, and then vote.
    """
    allowed_mods = _parse_allowed_modalities(args)
    max_tokens = 128
    n_rounds = getattr(args, 'debate_rounds', 3)
    
    # 1. Define Potential Agents
    potential_agents = [
        ('ehr', 'EHR Specialist', ['ehr'], "You are an EHR Specialist. Analyze the vitals."),
        ('cxr', 'CXR Specialist', ['cxr'], "You are a Chest X-ray Specialist. Analyze the image."),
        ('ps', 'Patient Summary Specialist', ['ps'], "You are a Clinical Historian. Analyze the patient summary."),
        ('rr', 'Radiology Report Specialist', ['rr'], "You are a Radiology Report Specialist. Analyze the text report.")
    ]
    
    # 2. Select Active Agents based on args.modalities
    agents = []
    for mod_key, name, mod_list, sys_prompt in potential_agents:
        if mod_key in allowed_mods:
            agents.append({'name': name, 'mods': mod_list, 'sys': sys_prompt})
            
    if not agents:
        return _format_batch_results(batch, ["No Agents"]*len(batch), np.zeros(len(batch)), [])

    current_arguments = [[] for _ in range(len(agents))] 
    
    # --- Round 0: Initial Arguments ---
    for k, agent in enumerate(agents):
        prompts = []
        for p in batch:
            content, _ = _build_prompt_content(p, p.get('ehr_text', ""), agent['mods'], prompt_type="standard")
            prompts.append([{"role": "user", "content": [{"type": "text", "text": f"{agent['sys']} State reasoning for mortality risk.\n\n"}] + content}])
        current_arguments[k] = generate_response(prompts, model, processor, max_tokens)

    # --- Rounds 1 to N: Debate Loop ---
    for r in range(1, n_rounds):
        new_arguments = [[] for _ in range(len(agents))]
        for k, agent in enumerate(agents):
            prompts = []
            for i in range(len(batch)):
                peer_txt = ""
                for j, peer in enumerate(agents):
                    if k != j: peer_txt += f"{peer['name']}: {current_arguments[j][i]}\n"
                
                content, _ = _build_prompt_content(batch[i], batch[i].get('ehr_text', ""), agent['mods'], prompt_type="standard")
                full_txt = f"{agent['sys']} \n\n--- COLLEAGUE OPINIONS ---\n{peer_txt}\n--- TASK ---\nUpdate your analysis based on colleagues.\nAnalysis:"
                prompts.append([{"role": "user", "content": [{"type": "text", "text": full_txt}] + content}])
            
            new_arguments[k] = generate_response(prompts, model, processor, max_tokens)
        current_arguments = new_arguments
        _print_debug_sample(args, batch, prompts, tag=f"Debate UniModal Round {r}")

    # --- Final Consensus: Average of Updated Beliefs ---
    all_agent_probs = []
    for k, agent in enumerate(agents):
        prompts = []
        for i in range(len(batch)):
            txt = f"Based on your final analysis: {current_arguments[k][i]}\nDoes this patient die? Answer Yes or No\nAnswer:"
            prompts.append([{"role": "user", "content": [{"type": "text", "text": txt}]}])
        _, probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
        all_agent_probs.append(probs)

    avg_probs = np.mean(all_agent_probs, axis=0)
    return _format_batch_results(batch, [f"Debate UniModal ({len(agents)} agents)"]*len(batch), avg_probs, [{'patient_summary': 1, 'ehr_timeseries': 1, 'radiology_report': 1, 'cxr': 1}] * len(batch))


def _run_debate_multimodal_batch(batch, model, processor, args):
    """
    Multimodal Debate: 4 Agents (Temp=0.7) argue for N rounds.
    """
    allowed_mods = _parse_allowed_modalities(args)
    max_tokens = getattr(args, 'max_new_tokens', 128)
    n_rounds = getattr(args, 'debate_rounds', 3)
    n_agents = 4 
    
    print(f"   [Debate-Multimodal] Running {n_rounds} rounds with {n_agents} agents...")

    # 1. Init
    base_prompts = []
    for p in batch:
        content, _ = _build_prompt_content(p, p.get('ehr_text', ""), allowed_mods, prompt_type="cot_reasoning")
        base_prompts.append([{"role": "user", "content": content}])
    
    current_args = [] # [Agent][Batch]
    for k in range(n_agents):
        current_args.append(generate_response(base_prompts, model, processor, max_tokens, do_sample=True, temperature=0.7, top_k=50))

    # 2. Loop
    for r in range(1, n_rounds):
        new_args = []
        for k in range(n_agents):
            prompts = []
            for i in range(len(batch)):
                peer_txt = ""
                for j in range(n_agents):
                    if k != j: peer_txt += f"Agent {j+1}: {current_args[j][i]}\n"
                
                # Re-inject data + peers
                content, _ = _build_prompt_content(batch[i], batch[i].get('ehr_text', ""), allowed_mods, prompt_type="standard")
                txt = f"--- PEER ANALYSES ---\n{peer_txt}\n--- TASK ---\nUpdate your analysis based on peers.\nAnalysis:"
                prompts.append([{"role": "user", "content": [{"type": "text", "text": txt}] + content}])
            
            new_args.append(generate_response(prompts, model, processor, max_tokens, do_sample=True, temperature=0.7, top_k=50))
        current_args = new_args
        _print_debug_sample(args, batch, prompts, tag=f"Debate MultiModal Round {r}")

    # 3. Final Vote
    all_probs = []
    for k in range(n_agents):
        prompts = []
        for i in range(len(batch)):
            txt = f"Final Decision based on: {current_args[k][i]}\nDie in ICU? Yes or No\nAnswer:"
            prompts.append([{"role": "user", "content": [{"type": "text", "text": txt}]}])
        _, probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    return _format_batch_results(batch, [f"Debate MM ({n_rounds} rds)"]*len(batch), avg_probs, [{'patient_summary': 1, 'ehr_timeseries': 1, 'radiology_report': 1, 'cxr': 1}] * len(batch))



def _run_dual_agent_batch(batch, model, processor, args):
    # Selector
    selector_prompts = []
    available_modalities = "EHR_TIMESERIES, RADIOLOGY_REPORTS, CXR_IMAGE"
    for p in batch:
        prompt_text = (
            "You are a clinical triage agent. Your task is to determine which *additional* data modalities are "
            "necessary to predict in-hospital mortality, based *only* on the following patient summary.\n"
            f"Patient Summary:\n{p['patient_summary_text']}\n\n"
            f"Which of the following *additional* modalities do you require? "
            f"Available: [{available_modalities}]\n"
            "Respond with a comma-separated list of the modalities you need (e.g., 'EHR_TIMESERIES, CXR_IMAGE'). "
            "If the summary is sufficient, respond with 'NONE'."
        )
        selector_prompts.append([{"role": "user", "content": [{"type": "text", "text": prompt_text}]}])
    selector_responses = generate_response(selector_prompts, model, processor, max_tokens=20)
    
    # Decision
    decision_prompts = []
    modality_requests_list = [] 
    
    for i, patient in enumerate(batch):
        selection_text = selector_responses[i].upper()
        req_ehr = "EHR_TIMESERIES" in selection_text
        req_report = "RADIOLOGY_REPORT" in selection_text
        req_cxr = "CXR_IMAGE" in selection_text
        
        modality_requests = {
            'patient_summary': 1, 'ehr_timeseries': 1 if req_ehr else 0,
            'radiology_report': 1 if req_report else 0, 'cxr': 0
        }
        
        current_mods = ['ps']
        if req_ehr: current_mods.append('ehr')
        if req_report: current_mods.append('rr')
        if req_cxr: current_mods.append('cxr')
        
        content, has_image = _build_prompt_content(
            patient, patient.get('ehr_text', ''), current_mods, is_training_example=False
        )
        if has_image: modality_requests['cxr'] = 1
        decision_prompts.append([{"role": "user", "content": content}])
        modality_requests_list.append(modality_requests)
    
    # --- DEBUGGING HOOK ---
    _print_debug_sample(args, batch, decision_prompts)

    texts, exp_probs = generate_yes_no_probability(decision_prompts, model, processor, max_tokens=1)
    return _format_batch_results(batch, texts, exp_probs, modality_requests_list)

def initialize_agent_setup(batch, args):
    model, processor = get_model_and_processor(args)
    agent_setup_name = args.agent_setup
    
    if agent_setup_name == 'SingleAgent':
        return _run_single_agent_batch(batch, model, processor, args)
    elif agent_setup_name == 'FewShot':
        return _run_few_shot_batch(batch, model, processor, args)
    elif agent_setup_name == 'SingleAgent-CoT':
        return _run_single_agent_cot_batch(batch, model, processor, args)
    elif agent_setup_name == 'SingleAgent-CoT-SC':
        return _run_single_agent_self_consistency_batch(batch, model, processor, args)
    elif agent_setup_name == 'SelfRefine':
        return _run_self_refine_batch(batch, model, processor, args)
    elif agent_setup_name == 'MultiAgent':
        return _run_multi_agent_batch(batch, model, processor, args)
    elif agent_setup_name == 'MajorityVote':
        return _run_majority_vote_uni_modal_batch(batch, model, processor, args)
    elif agent_setup_name == 'Debate_Unimodal':
        return _run_debate_unimodal_batch(batch, model, processor, args)
    elif agent_setup_name == 'Debate_Multimodal':
        return _run_debate_multimodal_batch(batch, model, processor, args)
    elif agent_setup_name == 'DualAgent':
        return _run_dual_agent_batch(batch, model, processor, args)
    else:
        raise ValueError(f"Unknown agent_setup: '{agent_setup_name}'.")