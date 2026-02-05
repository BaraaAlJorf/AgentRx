import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Imports from your architecture
from agent_architectures import (
    _build_prompt_content,
    generate_yes_no_probability,
    _format_batch_results,
    _parse_allowed_modalities,
    STANDARD_SYS_TEXT
)

# -------------------------------------------------------------------------
# 1. HELPER: PROBABILITY CALIBRATION
# -------------------------------------------------------------------------

def inverse_sigmoid(p, epsilon=1e-7):
    """Safely converts probability to logit."""
    p = np.clip(p, epsilon, 1.0 - epsilon)
    return np.log(p / (1.0 - p))

def calibrate_prob(prob, temperature):
    """
    Applies temperature scaling to a scalar probability.
    prob: float (0.0 to 1.0)
    temperature: float (T > 0)
    """
    if temperature == 1.0:
        return prob
    
    # 1. Convert to Logit
    logit = inverse_sigmoid(prob)
    
    # 2. Apply Temperature (T < 1 sharpens, T > 1 softens)
    scaled_logit = logit / temperature
    
    # 3. Convert back to Probability
    return 1.0 / (1.0 + np.exp(-scaled_logit))

# -------------------------------------------------------------------------
# 2. TRAINING PHASE (GRADIENT DESCENT WITH BEST CHECKPOINTING)
# -------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    def __init__(self, init_val=1.0):
        super().__init__()
        # Optimize log_temp to ensure T is always positive
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_val), dtype=torch.float32))

    def forward(self, logits):
        return logits / torch.exp(self.log_temperature)

def optimize_single_modality_gd(raw_probs, labels, mod_name, learning_rate=0.001, epochs=200):
    if not raw_probs: return 1.0

    # Prepare Data
    logits_tensor = torch.tensor(inverse_sigmoid(np.array(raw_probs)), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss()

    # --- 1. Measure Baseline (T=1.0) ---
    # We record the starting loss at T=1.0. 
    # If optimization fails or drifts, we fall back to this.
    with torch.no_grad():
        baseline_loss = criterion(logits_tensor, labels_tensor).item()
    
    best_temp = 1.0
    best_loss = baseline_loss

    # --- 2. Setup Optimization ---
    # Start optimization at T=1.5 (slightly smooth) to give gradients room to move
    model = TemperatureScaler(init_val=1.5)
    optimizer = optim.Adam([model.log_temperature], lr=learning_rate)

    model.train()
    pbar = tqdm(range(epochs), desc=f"   Optimizing {mod_name.upper()}", leave=False)
    
    for _ in pbar:
        optimizer.zero_grad()
        
        # Forward Pass
        loss = criterion(model(logits_tensor), labels_tensor)
        
        # Backward Pass
        loss.backward()
        optimizer.step()
        
        # --- 3. Best Checkpoint Check ---
        # We check loss at every step. If it's the best we've seen, we save T.
        current_loss = loss.item()
        
        if current_loss < best_loss:
            best_loss = current_loss
            # Detach the value so we save the float, not the graph
            best_temp = torch.exp(model.log_temperature).item()
            
        pbar.set_postfix({'loss': f"{current_loss:.4f}", 'best_T': f"{best_temp:.2f}"})

    print(f"   [{mod_name.upper()}] Best T: {best_temp:.4f} (Baseline Loss: {baseline_loss:.4f} -> Best Loss: {best_loss:.4f})")
    
    return best_temp

def run_calibration_training(model, processor, val_loader, args):
    """Phase 1: Validation Loop -> Learn Temperatures (No Filtering)"""
    print(f"\n[Calibration] Phase 1: Collecting Validation Data ({len(val_loader.dataset)} samples)...")
    
    configs = [
        ('ps', 'PS', "Predict mortality based ONLY on the summary."), 
        ('ehr', 'EHR', "Predict mortality based ONLY on the vitals."), 
        ('rr', 'RR', "Predict mortality based ONLY on the radiology report."), 
        ('cxr', 'CXR', "Predict mortality based ONLY on the chest X-ray.")
    ]
    
    collected_probs = {m_key: [] for m_key, _, _ in configs}
    collected_labels = {m_key: [] for m_key, _, _ in configs}
    allowed_mods = _parse_allowed_modalities(args)

    # --- Collection Loop ---
    for batch in tqdm(val_loader, desc="[Phase 1] Collecting Val Probs"):
        labels = [p['labels']['in_hospital_mortality_48hr'] for p in batch]
        
        for mod_key, mod_name, sys_desc in configs:
            if mod_key in allowed_mods:
                prompts = []
                # STRICT UNCONDITIONAL LOOP (Includes empty/missing data prompts)
                for p in batch:
                    content, _ = _build_prompt_content(p, p.get('ehr_text', ""), [mod_key], prompt_type="standard")
                    prompts.append([{"role": "user", "content": [{"type": "text", "text": f"{STANDARD_SYS_TEXT}\n{sys_desc}\n\n"}] + content}])
                
                _, probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
                
                collected_probs[mod_key].extend(probs)
                collected_labels[mod_key].extend(labels)

    # --- Optimization Loop ---
    print("\n[Calibration] Phase 2: Optimizing Temperatures (Gradient Descent)...")
    learned_temps = {}
    for mod_key, _, _ in configs:
        if mod_key in allowed_mods and len(collected_probs[mod_key]) > 0:
            learned_temps[mod_key] = optimize_single_modality_gd(collected_probs[mod_key], collected_labels[mod_key], mod_key, args.lr)
        else:
            learned_temps[mod_key] = 1.0
            
    return learned_temps

# -------------------------------------------------------------------------
# 3. INFERENCE PHASE (CUSTOM MAJORITY VOTE)
# -------------------------------------------------------------------------

def _run_calibrated_majority_vote_batch(batch, model, processor, args, temps):
    """
    Inference: EXACT structure from your request + Calibration injection.
    """
    allowed_mods = _parse_allowed_modalities(args)
    batch_size = len(batch)
    all_voter_probs = []
    active_voters = []
    
    # For debug printing logic (optional, keeping it clean for the return)
    per_modality_probs = {}

    configs = [
        ('ps', 'PS', "Predict mortality based ONLY on the summary."), 
        ('ehr', 'EHR', "Predict mortality based ONLY on the vitals."), 
        ('rr', 'RR', "Predict mortality based ONLY on the radiology report."), 
        ('cxr', 'CXR', "Predict mortality based ONLY on the chest X-ray.")
    ]
    
    for mod_key, mod_name, sys_desc in configs:
        if mod_key in allowed_mods:
            prompts = []
            
            # 1. Unconditional Prompt Build (Includes Empty Data)
            for p in batch:
                content, _ = _build_prompt_content(p, p.get('ehr_text', ""), [mod_key], prompt_type="standard")
                prompts.append([{"role": "user", "content": [{"type": "text", "text": f"{STANDARD_SYS_TEXT}\n{sys_desc}\n\n"}] + content}])
            
            # 2. Generate Raw Probabilities
            _, raw_probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
            
            # 3. Apply Calibration (Inject T)
            t = temps.get(mod_key, 1.0)
            calibrated_probs = np.array([calibrate_prob(p, t) for p in raw_probs])

            # 4. Store Calibrated Results
            all_voter_probs.append(calibrated_probs)
            active_voters.append(mod_name)
            per_modality_probs[mod_name] = calibrated_probs

    if not all_voter_probs: 
        return _format_batch_results(batch, ["No Data"]*batch_size, np.zeros(batch_size), [])
    
    # 5. Average
    avg_probs = np.mean(all_voter_probs, axis=0)
    
    final_texts = [f"Calibrated Vote ({'+'.join(active_voters)}) Temps:{temps}"] * batch_size
    reqs = [{'patient_summary': 1, 'ehr_timeseries': 1, 'radiology_report': 1, 'cxr': 1}] * batch_size
    
    return _format_batch_results(batch, final_texts, avg_probs, reqs)


def run_inference_loop(model, processor, loader, args, learned_temps):
    """Phase 3: Test Loop -> Apply Temperatures"""
    print(f"\n[Inference] Phase 3: Running Calibrated Majority Vote on Test Set ({len(loader.dataset)} samples)...")
    
    all_results = []
    for batch in tqdm(loader, desc="[Phase 3] Test Inference"):
        batch_res = _run_calibrated_majority_vote_batch(batch, model, processor, args, learned_temps)
        all_results.extend(batch_res)
        
    return all_results

# -------------------------------------------------------------------------
# 4. MAIN FLOW ORCHESTRATOR
# -------------------------------------------------------------------------

def run_calibrated_flow(model, processor, val_loader, test_loader, args):
    # Step 1: Learn T on Validation
    learned_temps = run_calibration_training(model, processor, val_loader, args)
    
    print("\n" + "="*40)
    print(f"CALIBRATION RESULT: {learned_temps}")
    print("="*40 + "\n")
    
    # Step 2: Apply T on Test
    results = run_inference_loop(model, processor, test_loader, args, learned_temps)
    
    return results