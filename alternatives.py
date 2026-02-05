def _run_single_agent_batch(batch, model, processor, args):
    # --- 1. DEFINE TARGETS ---
    target_patients = {
       (10001884, 37510196),
       (10002155, 33685454),
       (10010867, 39880770),
       (10013643, 33072499),
       (10014729, 33558396),
       (10018328, 31269608),
       (10019777, 34578020),
       (10032176, 35599569),
       (10063534, 34475789),
       (10078115, 36265401),
       (10082986, 33902924)
    }

    # --- 2. FILTER BATCH ---
    # Find which indices in the current batch match our target list
    target_indices = []
    for i, p in enumerate(batch):
        if (p.get('subject_id'), p.get('stay_id')) in target_patients:
            target_indices.append(i)

    # Prepare return containers for the FULL batch (filled with dummies initially)
    # This ensures the pipeline doesn't crash due to size mismatches.
    batch_size = len(batch)
    final_texts = ["Skipped (Debug)"] * batch_size
    final_probs = np.zeros(batch_size, dtype=np.float32)
    final_modality_requests = [{'patient_summary': 0, 'ehr_timeseries': 0, 'radiology_report': 0, 'cxr': 0} for _ in range(batch_size)]

    # Optimization: If no target patients are in this batch, return immediately
    if not target_indices:
        return _format_batch_results(batch, final_texts, final_probs, final_modality_requests)

    # --- 3. ABLATION LOOP (RUNNING ON SUBSET ONLY) ---
    ablation_steps = [
        (['ps'], "PS Only"),
        (['ps', 'ehr'], "PS + EHR"),
        (['ps', 'ehr', 'rr'], "PS + EHR + RR"),
        (['ps', 'ehr', 'rr', 'cxr'], "PS + EHR + RR + CXR")
    ]
    
    # Create the subset of patient objects once
    target_batch_subset = [batch[i] for i in target_indices]

    for current_mods, step_name in ablation_steps:
        prompts = []
        
        # Build prompts ONLY for the filtered subset
        for patient in target_batch_subset:
            ehr_text = patient.get('ehr_text', "EHR Data Not Available")
            content, has_image = _build_prompt_content(
                patient, ehr_text, current_mods, is_training_example=False, prompt_type="standard"
            )
            system_msg = {"type": "text", "text": "You are an expert ICU risk prediction model. This patient was just admitted.\n\n"}
            prompts.append([{"role": "user", "content": [system_msg] + content}])

        # Inference on subset
        texts, exp_probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)

        # --- PRINTING & STORING RESULTS ---
        for idx_in_subset, real_batch_idx in enumerate(target_indices):
            # idx_in_subset: 0, 1, 2... (index in the small prompts list)
            # real_batch_idx: 12, 45, 60... (index in the original batch)
            
            p = batch[real_batch_idx]
            curr_sub = p.get('subject_id')
            curr_stay = p.get('stay_id')
            
            prob = float(exp_probs[idx_in_subset])
            pred = "YES" if prob >= 0.5 else "NO"

            # Print Header only on first step
            if step_name == "PS Only":
                print(f"\n=== Patient {real_batch_idx} [DEBUG TARGET] (Sub: {curr_sub}, Stay: {curr_stay}) ===")
            
            print(f"   [{step_name:<20}] Prob: {prob:.4f} | Pred: {pred}")

            # If this is the LAST step (Full Modality), save results to the return arrays
            if step_name == "PS + EHR + RR + CXR":
                print("-" * 50) # Footer
                final_texts[real_batch_idx] = texts[idx_in_subset]
                final_probs[real_batch_idx] = exp_probs[idx_in_subset]
                
                # Check actual availability for the request log
                _, has_image_check = _build_prompt_content(p, p.get('ehr_text', ""), ['cxr'], prompt_type="standard")
                final_modality_requests[real_batch_idx] = {
                    'patient_summary': 1,
                    'ehr_timeseries': 1,
                    'radiology_report': 1,
                    'cxr': 1 if has_image_check else 0
                }

    # Return results (Targets have real data, non-targets have dummy data)
    return _format_batch_results(batch, final_texts, final_probs, final_modality_requests)
###GATEKEEPER ARCHITECTURE AGENTICDS

def _run_agenticds_batch(batch, model, processor, args):
    """
    AgentiCDS (Gatekeeper Architecture):
    1. Gatekeeper Agent: Reviews ALL data against Rules -> Selects relevant modalities.
    2. Judge Agent: Uses the STANDARD prompt structure (same as SingleAgent), but only sees the filtered data.
    """
    # --- 0. Setup ---
    if not hasattr(args, 'rulebook_dir'):
        args.rulebook_dir = args.output_dir

    _, diagnostic_rules = _load_rulebooks(args.rulebook_dir)
    allowed_mods = _parse_allowed_modalities(args)
    max_tokens = getattr(args, 'max_new_tokens', 100)

    # --- 1. Prepare Rules for Gatekeeper ---
    def format_rule_entry(r):
        feat = r.get("Feature of Interest", "Feature")
        pat = r.get("Patterns", "")
        rule = r.get("Rule", "")
        return f"- **{feat}**: {rule} (Look for: {pat})"

    rule_context = []
    # Multimodal
    if 'multimodal' in diagnostic_rules:
        rule_context.append("### MULTIMODAL RULES ###")
        for r in diagnostic_rules['multimodal']:
            rule_context.append(format_rule_entry(r))
    # Unimodal
    for m in ['ehr', 'cxr', 'rr', 'ps']:
        if m in allowed_mods and m in diagnostic_rules:
            mod_name = m.upper()
            rule_context.append(f"\n### {mod_name} RULES ###")
            for r in diagnostic_rules[m]:
                rule_context.append(format_rule_entry(r))

    rules_str = "\n".join(rule_context)

    # --- STEP 1: The Gatekeeper (Selection Phase) ---
    print(f"   [AgentiCDS] Step 1: Selecting relevant modalities...")
    gate_prompts = []

    for p in batch:
        # Gatekeeper sees ALL available data to make the decision
        content, _ = _build_prompt_content(p, p.get('ehr_text', ''), allowed_mods, prompt_type="data_only")

        sys_header = (
            f"{STANDARD_SYS_TEXT} You are the Clinical Data Gatekeeper.\n"
            f"Review the patient data below against the Rulebook.\n\n"
            f"--- RULEBOOK ---\n"
            f"{rules_str}\n\n"
            f"--- PATIENT DATA ---\n"
        )
        
        sys_footer = (
            f"\n\n--- TASK ---\n"
            f"Which modalities contain evidence that triggers the rules above or indicates mortality risk?\n"
            f"Options: EHR (Electronic Health REcords), CXR (Chest X-ray Image), RR (Radiology Reports), PS (Patient Summary and History).\n"
            f"Select ONLY the useful ones. If a modality is normal/irrelevant, exclude it.\n\n"
            f"--- OUTPUT FORMAT ---\n"
            f"Respond strictly with the comma-separated codes.\n"
            f"Example: SELECTION: EHR, CXR, RR, PS\n"
            f"SELECTION:"
        )

        full_content = [{"type": "text", "text": sys_header}] + content + [{"type": "text", "text": sys_footer}]
        gate_prompts.append([{"role": "user", "content": full_content}])

    # Generate Selections
    gate_responses = generate_response(gate_prompts, model, processor, max_tokens=max_tokens)
    _print_debug_sample(args, batch, gate_prompts, tag="AgentiCDS Selection")
    print("Requested:",gate_responses)
    # --- STEP 2: The Judge (Standard Prediction on Filtered Data) ---
    print(f"   [AgentiCDS] Step 2: Judge Prediction on filtered data...")
    judge_prompts = []
    modality_requests_list = []
    
    for i, p in enumerate(batch):
        raw_response = gate_responses[i].upper()
        
        # 1. Parse Selection
        selected_mods = []
        if "EHR" in raw_response: selected_mods.append('ehr')
        if "CXR" in raw_response: selected_mods.append('cxr')
        if "RR" in raw_response:  selected_mods.append('rr')
        if "PS" in raw_response:  selected_mods.append('ps')
        
        # Fallback: if empty, default to PS so the model isn't blind
        if not selected_mods: selected_mods = ['ps']

        # 2. Build STANDARD Prompt
        # This function handles the "Answer: " trigger automatically when prompt_type="standard"
        content_filtered, has_image = _build_prompt_content(p, p.get('ehr_text', ''), selected_mods, prompt_type="standard")
        
        # 3. Construct Final Prompt
        # We use STANDARD_SYS_MSG to keep it identical to the baseline SingleAgent
        judge_prompts.append([{"role": "user", "content": [STANDARD_SYS_MSG] + content_filtered}])

        # Logging
        modality_requests_list.append({
            'patient_summary': 1 if 'ps' in selected_mods else 0,
            'ehr_timeseries': 1 if 'ehr' in selected_mods else 0,
            'radiology_report': 1 if 'rr' in selected_mods else 0,
            'cxr': 1 if 'cxr' in selected_mods and has_image else 0
        })

    # Generate Final Decision
    final_texts, final_probs = generate_yes_no_probability(judge_prompts, model, processor, max_tokens=1)
    print(final_texts)
    # Format Output
    combined_logs = []
    for i in range(len(batch)):
        log = f"[Gatekeeper Selection]: {gate_responses[i]}\n[Judge Decision]: {final_texts[i]}"
        combined_logs.append(log)

    return _format_batch_results(batch, combined_logs, final_probs, modality_requests_list)
    
#### GENERATOR - VERIFIER
def _run_agenticds_batch(batch, model, processor, args):
    """
    AgentiCDS 2.0: Generator-Verifier Architecture (Rich Rules Edition).
    
    Concept:
    - Step 1 (Generator): Multimodal Agent sees ALL data -> Generates Clinical Findings & Draft Prediction.
    - Step 2 (Verifier): Multimodal Agent sees Raw Data + Draft + Detailed Rules -> Validates -> Decides.
    """
    # --- 0. Setup ---
    if not hasattr(args, 'rulebook_dir'):
        args.rulebook_dir = args.output_dir

    # Load rules and parse allowed modalities
    _, diagnostic_rules = _load_rulebooks(args.rulebook_dir)
    allowed_mods = _parse_allowed_modalities(args)
    max_tokens = getattr(args, 'max_new_tokens', 300)

    # --- 1. Prepare Rich Rules Context ---
    def format_rule_entry(r):
        feat = r.get("Feature of Interest", "Feature")
        desc = r.get("Description", "")
        pat = r.get("Patterns", "")
        rule = r.get("Rule", "")
        
        entry = f"- **{feat}**"
        if desc: entry += f"\n  * Context: {desc}"
        if pat:  entry += f"\n  * Critical Pattern: {pat}"
        if rule: entry += f"\n  * ACTION RULE: {rule}"
        return entry

    rule_context = []
    
    # A. Multimodal Rules (Highest Priority)
    if 'multimodal' in diagnostic_rules:
        rule_context.append("### MULTIMODAL COMBINATION GUIDELINES ###")
        for r in diagnostic_rules['multimodal']:
            rule_context.append(format_rule_entry(r))

    # B. Unimodal Rules (Support)
    for m in ['ehr', 'cxr', 'rr', 'ps']:
        if m in allowed_mods and m in diagnostic_rules:
            mod_name = m.upper()
            rule_context.append(f"\n### {mod_name} SPECIFIC GUIDELINES ###")
            for r in diagnostic_rules[m]:
                rule_context.append(format_rule_entry(r))

    rules_str = "\n".join(rule_context)

    # --- STEP 2: The Generator (Multimodal Analysis) ---
    print(f"   [AgentiCDS 2.0] Step 1: Generator Analysis...")
    gen_prompts = []
    modality_requests_list = []

    for p in batch:
        # Build Multimodal Content (Raw Data)
        content, has_image = _build_prompt_content(p, p.get('ehr_text', ''), allowed_mods, prompt_type="data_only")

        modality_requests_list.append({
            'patient_summary': 1 if 'ps' in allowed_mods else 0,
            'ehr_timeseries': 1 if 'ehr' in allowed_mods else 0,
            'radiology_report': 1 if 'rr' in allowed_mods else 0,
            'cxr': 1 if 'cxr' in allowed_mods and has_image else 0
        })

        # System Prompt
        sys_gen = (
            f"{STANDARD_SYS_TEXT}\n"
            f"--- TASK ---\n"
            f"Analyze the provided multimodal patient data.\n"
            f"1. Extract key clinical findings from EACH available modality.\n"
            f"2. Synthesize these findings into a rationale for ICU mortality risk.\n"
            f"3. Provide a Draft Prediction.\n\n"
            f"--- PATIENT DATA ---\n" # Data comes immediately after this
        )
        
        # Instruction Footer
        gen_footer = (
            f"\n\n--- OUTPUT FORMAT ---\n"
            f"Findings: [List key abnormalities per modality]\n"
            f"Rationale: [Synthesis]\n"
            f"Draft Prediction: [Yes/No]"
        )

        # Structure: [Header] + [Data] + [Footer]
        full_content = [{"type": "text", "text": sys_gen}] + content + [{"type": "text", "text": gen_footer}]
        gen_prompts.append([{"role": "user", "content": full_content}])

    # Generate Drafts
    draft_analyses = generate_response(gen_prompts, model, processor, max_tokens=max_tokens)
    _print_debug_sample(args, batch, gen_prompts, tag="AgentiCDS Gen Step")


    # --- STEP 3: The Verifier (Rule Abider) ---
    print(f"   [AgentiCDS 2.0] Step 2: Rule Verification...")
    ver_prompts = []

    for i, p in enumerate(batch):
        # We RE-USE the raw data content
        content, _ = _build_prompt_content(p, p.get('ehr_text', ''), allowed_mods, prompt_type="data_only")
        draft = draft_analyses[i]

        # 1. Header: Role + Rules + Data Intro
        sys_header = (
            f"{STANDARD_SYS_TEXT} You are the Clinical Rule Verifier.\n\n"
            f"--- DIAGNOSTIC RULEBOOK ---\n"
            f"{rules_str}\n\n"
            f"--- INPUT 1: RAW PATIENT DATA ---\n"
        )
        
        # 2. Footer: Generator Analysis + Task + Final Trigger
        sys_footer = (
            f"\n\n--- INPUT 2: GENERATOR ANALYSIS ---\n"
            f"{draft}\n\n"
            f"--- TASK ---\n"
            f"1. Verify: Do the Generator's findings actually exist in the Raw Data above? (Ignore hallucinations).\n"
            f"2. Apply Rules: Check if the *verified* findings match any 'Critical Pattern' listed in the Rulebook.\n"
            f"3. Decide: Does this patient die in the ICU? (You MUST follow the 'ACTION RULE' if a pattern is matched).\n\n"
            f"Answer only using one word - Yes or No\n"
            f"Answer:"
        )

        # Correct Structure: [Header] -> [Images/Text Data] -> [Footer with Question]
        full_content = [{"type": "text", "text": sys_header}] + content + [{"type": "text", "text": sys_footer}]
        ver_prompts.append([{"role": "user", "content": full_content}])

    # Generate Final Decision
    final_texts, final_probs = generate_yes_no_probability(ver_prompts, model, processor, max_tokens=1)
    print("Answer:",final_texts)
    # Format Output
    combined_texts = []
    for i in range(len(batch)):
        log = f"[Generator Draft]:\n{draft_analyses[i]}\n\n[Verifier Final]:\n{final_texts[i]}"
        combined_texts.append(log)

    return _format_batch_results(batch, combined_texts, final_probs, modality_requests_list)

###VANILLA:
def _run_agenticds_batch(batch, model, processor, args):
    """
    Inference Phase: Experts (w/ JSON Rules) -> Judge (w/ JSON Rules).
    Always uses the Judge for final decisions, regardless of modality count.
    Matches availability strings from the dataset generation script.
    """
    if not hasattr(args, 'rulebook_dir'): 
        args.rulebook_dir = args.output_dir
        
    _, diagnostic_rules = _load_rulebooks(args.rulebook_dir)
    allowed_mods = _parse_allowed_modalities(args)
    batch_size = len(batch)
    
    # Map abbreviations to full names used in the data generation script
    modality_full_names = {
        'ehr': 'Electronic Health Records',
        'cxr': 'Chest X-ray Image',
        'rr': 'Radiology Reports',
        'ps': 'Patient Summary'
    }
    
    # 1. Expert Analysis Phase
    expert_analyses = ["" for _ in range(batch_size)]
    modality_usage_tracking = [set() for _ in range(batch_size)]
    
    for mod in ['ehr', 'cxr', 'rr', 'ps']:
        if mod not in allowed_mods: 
            continue
        
        full_name = modality_full_names[mod]
        mod_rule_list = diagnostic_rules.get(mod, [])
        
        # Format rules for this modality
        formatted_rules = []
        for r in mod_rule_list:
            feat = r.get("Feature of Interest", "Unknown")
            rule_text = r.get("Rule", "No specific rule")
            pat = r.get("Patterns", "N/A")
            formatted_rules.append(f"- Feature: {feat}\n  Pattern: {pat}\n  Rule: {rule_text}")
        
        mr_str = "\n".join(formatted_rules) if formatted_rules else "Analyze the clinical data for mortality signs."
        
        # Identify active patients for this specific modality
        active_indices = []
        prompts = []
        
        for i, p in enumerate(batch):
            is_available = False
            if mod == 'ehr' and p.get('ehr_timeseries_path') not in [None, '', 'N/A']:
                is_available = True
            elif mod == 'cxr' and p.get('cxr_image_path') != "CXR not available":
                is_available = True
            elif mod == 'rr' and p.get('radiology_report_text') != "Radiology report not available":
                is_available = True
            elif mod == 'ps' and "PATIENT BASELINE SUMMARY" in p.get('patient_summary_text', ''):
                is_available = True
            
            if is_available:
                active_indices.append(i)
                modality_usage_tracking[i].add(mod)
                sys = (
                    f"You are an expert specialist in {full_name}.\n"
                    f"Apply these clinical diagnostic rules to the attached {full_name} data collected during the first 48 hours of this patient's ICU stay:\n{mr_str}\n\n"
                    f"Task: Analyze the attached {full_name} and provide a concise clinical assessment addressing whether this patient is likely to die in the ICU before their stay is over.\n"
                    "You MUST output exactly one valid JSON object and nothing else (no markdown, no extra text) where each object has exactly these keys:\n"
                    f'   - "Modality", "{full_name}"\n'
                    f'   - "Findings of Interest": short description of findings relevant to the attached modality\n'
                    f'   - "Vote": this patient dies before ICU discharge or this patient does not die before ICU discharge\n'
                    f'   - "Confidence": how confident are you in your vote?\n'
                    f"Output only valid JSON (an array). No extra commentary or surrounding text. Keep entries concise.\n"
                )
                content, _ = _build_prompt_content(p, p.get('ehr_text', ''), [mod], prompt_type="data_only")
                prompts.append([{"role": "user", "content": [{"type": "text", "text": sys}] + content}])

        # Batch call for the modality
        if prompts:
            outputs = generate_response(prompts, model, processor, max_tokens=300)
            for idx, i in enumerate(active_indices):
                expert_analyses[i] += f"[{full_name} Expert Report]:\n{outputs[idx]}\n\n"

    # 2. Multimodal Judge Synthesis (Always the final decision maker)
    mm_rule_list = diagnostic_rules.get('multimodal', [])
    mmr_str = "\n".join([f"- {r.get('Feature of Interest')}: {r.get('Rule')}" for r in mm_rule_list])
    
    final_prompts = []
    for i, p in enumerate(batch):
        reports = expert_analyses[i] if expert_analyses[i] else "No specialist data was available for this patient."
        # print("Reports:", reports)
        sys_header = (
            f"{STANDARD_SYS_TEXT} You are the Lead Diagnostician.\n"
            f"Cross-Modality Reasoning Rules:\n{mmr_str}\n\n"
            f"=== HOW TO USE RULES ===\n"
            f"- The multimodal rules above are provided exactly as 'Feature: Rule'.\n"
            f"- Use them to reconcile conflicts between modalities and prioritize the most reliable evidence.\n\n"
            f"--- INSTRUCTIONS ---\n"
            f"1. Review the raw 'Data' provided below.\n"
            f"2) Review the SPECIALIST JSON outputs below.\n"
            f"3) Accept specialist claims ONLY if supported by raw data.\n"
            f"4) Apply multimodal rules to decide ICU mortality.\n\n"
            f"Data:\n" 
        )
        
        # C. Get the Raw Data Content (Images + Text)
        # prompt_type="data_only" returns the images and the raw text strings without extra instruction wrappers
        active_mods = list(modality_usage_tracking[i])
        content_raw_data, _ = _build_prompt_content(p, p.get('ehr_text', ''), active_mods, prompt_type="data_only")
        
        # D. Prepare the Analyses & Decision Block
        # We append this *after* the raw data content
        analyses_and_decision = (
            f"\n\SPECIALIST JSON OUTPUTS:\n"
            f"{reports}\n"
            f"--- DECISION ---\n"
            f"Based on the Data and Analyses above, does this patient die in the ICU? Answer only using one word - Yes or No\n"
            f"Answer: "
        )
        
        # E. Construct the full message list
        # Structure: [System+DataHeader] + [Images/RawText] + [Analyses+Question]
        full_content = [{"type": "text", "text": sys_header}] + content_raw_data + [{"type": "text", "text": analyses_and_decision}]
        
        final_prompts.append([{"role": "user", "content": full_content}])
    #     # Case where no data was found for a patient
    #     reports = expert_analyses[i] if expert_analyses[i] else "No specialist data was available for this patient."
        
    #     sys_judge = (
    #         f"{STANDARD_SYS_TEXT} You are the Lead Diagnostician.\n"
    #         f"Cross-Modality Reasoning Rules:\n{mmr_str}\n\n"
    #         f"--- SPECIALIST ANALYSES ---\n{reports}\n\n"
    #         f"--- TASK ---\n"
    #         f"Review the specialist reports provided above."
    #         "--- DECISION ---\n"
    #         "Based on the analysis above, does this patient die in the ICU? Answer only using one word - Yes or No\n"
    #         "Answer: "
    #     )
        
    #     # Pass all active modalities to the judge for context
    #     active_mods = list(modality_usage_tracking[i])
    #     content_judge, _ = _build_prompt_content(p, p.get('ehr_text', ''), active_mods, prompt_type="data_only")
    #     final_prompts.append([{"role": "user", "content": [{"type": "text", "text": sys_judge}] + content_judge}])

    # # Batch calculation of probabilities for the Judge's final answers
    texts, probs = generate_yes_no_probability(final_prompts, model, processor, max_tokens=1)
    # Format results
    print('Answer:', texts)
    modality_reqs = []
    for usage in modality_usage_tracking:
        modality_reqs.append({
            'patient_summary': 1 if 'ps' in usage else 0,
            'ehr_timeseries': 1 if 'ehr' in usage else 0,
            'radiology_report': 1 if 'rr' in usage else 0,
            'cxr': 1 if 'cxr' in usage else 0
        })

    return _format_batch_results(batch, expert_analyses, probs, modality_reqs)
    
# def _run_agenticds_simplified_batch(batch, model, processor, args):
#     """
#     AgentiCDS Simplified (Single-Pass):
#     1. Loads ALL rules (Unimodal + Multimodal).
#     2. Feeds Raw JSON Rules + All Data into a single prompt.
#     3. Asks the standard mortality question.
#     """
#     if not hasattr(args, 'rulebook_dir'): 
#         args.rulebook_dir = args.output_dir
        
#     _, diagnostic_rules = _load_rulebooks(args.rulebook_dir)
#     allowed_mods = _parse_allowed_modalities(args)
    
#     # --- 1. Filter Rules for Active Modalities ---
#     # We only include rules for modalities we are actually using, plus multimodal rules.
#     active_rules = {}
    
#     # Add Unimodal Rules
#     for m in ['ehr', 'cxr', 'rr', 'ps']:
#         if m in allowed_mods and m in diagnostic_rules:
#             active_rules[m] = diagnostic_rules[m]
            
#     # Add Multimodal Rules (always included)
#     if 'multimodal' in diagnostic_rules:
#         active_rules['multimodal'] = diagnostic_rules['multimodal']

#     # Convert to Raw JSON String
#     all_rules_json = json.dumps(active_rules, indent=2)

#     # --- 2. Build Prompts ---
#     final_prompts = []
#     modality_requests_list = []

#     for i, p in enumerate(batch):
#         # Build prompt content with ALL allowed modalities included
#         content_raw_data, has_image = _build_prompt_content(
#             p, p.get('ehr_text', ''), allowed_mods, prompt_type="data_only"
#         )

#         # Construct the Unified System Prompt with Raw JSON
#         sys_header = (
#             f"{STANDARD_SYS_TEXT}\n"
#             f"--- DIAGNOSTIC RULEBOOK (JSON) ---\n"
#             f"{all_rules_json}\n\n"
#             f"--- INSTRUCTIONS ---\n"
#             f"1. Review the patient data provided below.\n"
#             f"2. Apply the JSON Diagnostic Rules strictly to the data where applicable.\n"
#             f"3. Predict In-Hospital Mortality.\n\n"
#             f"--- PATIENT DATA ---\n" 
#         )

#         # Construct the Question/Decision Block
#         decision_block = (
#             f"\n\n--- DECISION ---\n"
#             f"Based on the Data and Rules above, does this patient die in the ICU? Answer only using one word - Yes or No\n"
#             f"Answer: "
#         )

#         # Combine: Header + [Images/Data] + Question
#         full_content = [{"type": "text", "text": sys_header}] + content_raw_data + [{"type": "text", "text": decision_block}]
#         final_prompts.append([{"role": "user", "content": full_content}])

#         # Track requests for logging
#         modality_requests_list.append({
#             'patient_summary': 1 if 'ps' in allowed_mods else 0,
#             'ehr_timeseries': 1 if 'ehr' in allowed_mods else 0,
#             'radiology_report': 1 if 'rr' in allowed_mods else 0,
#             'cxr': 1 if 'cxr' in allowed_mods and has_image else 0
#         })

#     # --- 3. Inference ---
#     texts, probs = generate_yes_no_probability(final_prompts, model, processor, max_tokens=1)
    
#     return _format_batch_results(batch, texts, probs, modality_requests_list)
def _run_agenticds_batch(batch, model, processor, args):
    """
    AgentiCDS (Natural Language / Full Context Rules):
    1. Parses the JSON rulebook into detailed blocks (Feature, Description, Pattern, Rule).
    2. Specialist Agents apply Unimodal Rules -> Summaries.
    3. Judge Agent applies Multimodal Rules -> Final Decision.
    """
    if not hasattr(args, 'rulebook_dir'): 
        args.rulebook_dir = args.output_dir
        
    _, diagnostic_rules = _load_rulebooks(args.rulebook_dir)
    allowed_mods = _parse_allowed_modalities(args)
    batch_size = len(batch)
    
    # Mapping to give agents clear personas
    modality_names = {
        'ehr': 'Electronic Health Records Expert',
        'cxr': 'Chest X-ray Expert',
        'rr': 'Radiology Reports Expert',
        'ps': 'Patient Summary Expert'
    }

    # --- HELPER: Rule Formatter ---
    # Now includes ALL fields (Description, Patterns, Rule) for maximum context
    def format_rules(rule_list):
        if not rule_list: return "No specific guidelines."
        blocks = []
        for r in rule_list:
            # Extract all fields safely
            feat = r.get("Feature of Interest", "General Feature")
            desc = r.get("Description", "N/A")
            pat = r.get("Patterns", "N/A")
            rule = r.get("Rule", "")
            
            # Format as a distinct block for the model
            block = (
                f"- **FEATURE**: {feat}\n"
                f"  * Description: {desc}\n"
                f"  * Critical Pattern: {pat}\n"
                f"  * General RULE: {rule}"
            )
            blocks.append(block)
        return "\n\n".join(blocks)
    
    # --- PHASE 1: Specialist Analysis ---
    expert_reports = ["" for _ in range(batch_size)]
    modality_usage_tracking = [set() for _ in range(batch_size)]

    for mod in ['ehr', 'cxr', 'rr', 'ps']:
        if mod not in allowed_mods: continue
        
        # 1. Get Detailed Natural Language Rules for this modality
        rules_block = format_rules(diagnostic_rules.get(mod, []))
        
        # 2. Identify active patients for this modality
        active_indices = []
        prompts = []
        
        for i, p in enumerate(batch):
            # Check availability
            is_available = False
            if mod == 'ehr' and p.get('ehr_text'): is_available = True
            elif mod == 'cxr' and p.get('cxr_image_path') != "CXR not available": is_available = True
            elif mod == 'rr' and p.get('radiology_report_text') != "Radiology report not available": is_available = True
            elif mod == 'ps' and p.get('patient_summary_text'): is_available = True
            
            if is_available:
                active_indices.append(i)
                modality_usage_tracking[i].add(mod)
                
                content, _ = _build_prompt_content(p, p.get('ehr_text', ''), [mod], prompt_type="data_only")
                
                # --- Prompt parts ---

                preamble = (
                    f"You are the {modality_names[mod]}.\n"
                    f"This patient was just admitted to the ICU and is being attended to by a team of clinicians. Is this patient likely to die before discharge?\n"
                    f"Review the patient data below against these specific Risk Guidelines:\n\n"
                    f"{rules_block}\n\n"
                    f"--- PATIENT DATA ---\n"
                )
                
                # final_instructions = (
                #     f"\n--- TASK ---\n"
                #     f"1. Scan the data for the 'Critical Patterns' described above.\n"
                #     f"2. Use the corresponding 'General RULES' as heuristic guides to focus your attention, not as strict if–then statements.\n"
                #     f"   These rules often describe conditions that are treatable and may or may not progress depending on interventions.\n"
                #     f"   Interpret them as highlighting areas of risk that, if untreated or refractory, could contribute to the outcome described.\n"
                #     f"3. Integrate all available evidence to form a holistic clinical assessment rather than a rule-based verdict.\n\n"
                #     f"4. Summarize the patient's current state and the most likely outcome before discharge, accounting for both risk factors and potential responsiveness to treatment.\n"
                #     f"\n--- STRICT OUTPUT CONSTRAINTS (MUST FOLLOW) ---\n"
                #     f"- Output must be a SINGLE sentence.\n"
                #     f"- Use AT MOST THREE clinical findings or risk factors. NEVER include a fourth.\n"
                #     f"- Do NOT use bullet points, numbering, headings, or lists.\n"
                #     f"- Do NOT restate the guidelines or rules; describe only patient-specific findings and their clinical implications.\n"
                #     f"- Do NOT add extra commentary, disclaimers, or meta-explanations.\n\n"
                #     f"Begin your response directly with the memo text.\n"
                #     f"Memo:"
                # )
                
                final_instructions = (
                    f"\n--- TASK ---\n"
                    f"1. Scan the data for adverse or high-risk 'Critical Patterns' that increase the likelihood of poor outcome.\n"
                    f"2. Scan the data for reassuring, stabilizing, or improving patterns that decrease risk or suggest effective treatment.\n"
                    f"3. Write a clinical memo consisting of exactly two sentences:\n"
                    f"   - The FIRST sentence must summarize the most important adverse trends and risk factors, or explicitly state that none are present.\n"
                    f"   - The SECOND sentence must summarize the most important reassuring or improving trends, or explicitly state that none are present.\n"
                    f"  Ground both sentences in concrete patient features from the data and, when appropriate, the guidelines above.\n\n"
                    f"--- STRICT OUTPUT CONSTRAINTS (MUST FOLLOW) ---\n"
                    f"- Output must be a SINGLE paragraph containing EXACTLY two sentences. NEVER write a third sentence.\n"
                    f"- Do NOT label the sentences (do not write 'First sentence', 'Second sentence'). Just write the text.\n"
                    f"- Do NOT use bullet points, numbering, headings, or lists.\n"
                    f"- Do NOT restate the guidelines or rules; describe only patient-specific findings and their implications.\n"
                    f"- Do NOT add any extra commentary, meta-explanations, or disclaimers.\n"
                    f"- If you find no clear adverse trends, the FIRST sentence must state that explicitly.\n"
                    f"- If you find no clear reassuring trends, the SECOND sentence must state that explicitly.\n\n"
                    f"Begin your response directly with the memo text.\n"
                    f"Memo:"
                )


                
                # --- Assemble prompt in correct order ---
                prompts.append([
                    {
                        "role": "user",
                        "content": (
                            [{"type": "text", "text": preamble}]
                            + content
                            + [{"type": "text", "text": final_instructions}]
                        )
                    }
                ])


        # 3. Generate Reports
        if prompts:
            # Increased max_tokens slightly to allow for the extra detail
            outputs = generate_response(prompts, model, processor, max_tokens=1000)
            print(outputs)
            for idx, i in enumerate(active_indices):
                expert_reports[i] += f"--- {modality_names[mod]} Findings ---\n{outputs[idx]}\n\n"

    # --- PHASE 2: The Judge (Synthesis) ---
    # --- PHASE 2: The Judge (Synthesis) ---

    mm_rules_block = format_rules(diagnostic_rules.get('multimodal', []))
    final_prompts = []
    
    for i, p in enumerate(batch):
        dossier = expert_reports[i] if expert_reports[i] else "No specialist data available."
    
        # Build multimodal patient content for the judge
        # Use the same helper; include all modalities that were actually used
        used_mods = list(modality_usage_tracking[i]) or ['ps', 'ehr', 'rr', 'cxr']
        judge_content, _ = _build_prompt_content(
            p,
            p.get('ehr_text', ''),
            used_mods,
            prompt_type="data_only"
        )
    
        # Preamble + rules + specialist reports
        judge_preamble = (
            f"{STANDARD_SYS_TEXT} You are the Lead Diagnostician.\n"
            f"You must determine whetehr this patient dies before they are discharged from the ICU based on both the specialist reports and the original patient data.\n\n"
            f"--- IMPORTANT PRIOR KNOWLEDGE ---\n"
            f"- In typical ICU populations, most patients survive to discharge.\n"
            f"- Mortality is relatively uncommon compared with survival.\n"
            f"- Therefore, your default answer should be 'No' (patient survives) unless there is strong evidence otherwise.\n\n"
            f"--- COMPLEX INTERACTION GUIDELINES ---\n"
            f"Use these rules to resolve conflicts or combine evidence across modalities:\n"
            f"{mm_rules_block}\n\n"
            f"--- SPECIALIST REPORTS (SUMMARIES) ---\n"
            f"{dossier}\n\n"
            f"--- RAW PATIENT DATA (FOR VERIFICATION AND CONTEXT) ---\n"
            f"Review the following data to confirm or revise the impressions from the specialist reports.\n"
        )
    
        # Final decision instructions
        # judge_final_instructions = (
        #     f"\n--- DECISION RULE ---\n"
        #     f"Answer 'Yes' (patient dies in the ICU) ONLY if:\n"
        #     f"- There is clear, strong, and consistent evidence of severe, ongoing instability or multiorgan failure,\n"
        #     f"- AND the condition appears refractory to usual treatment or rapidly worsening,\n"
        #     f"- AND multiple modalities (when available) support a poor prognosis.\n"
        #     f"In all other situations (including weak, ambiguous, or single-modality risk signals), answer 'No' (patient survives).\n"
        #     f"- Avoid overcalling mortality: when in doubt, favor 'No'.\n\n"
        #     f"--- FINAL TASK ---\n"
        #     f"Based on the specialist reports, the raw patient data, and the decision rule above, does this patient die in the ICU?\n"
        #     f"Answer using exactly one word: Yes or No.\n"
        #     f"Answer:"
        # )
        
        judge_final_instructions = (
            f"\n--- CONSISTENCY CHECK (MUST PERFORM BEFORE DECIDING) ---\n"
            f"Before making a final decision, explicitly consider:\n"
            f"- Are there clear signs of clinical improvement or stability?\n"
            f"- Are the adverse trends persistent and refractory to treatment, or are they responding to interventions?\n"
            f"- Do multiple modalities consistently indicate severe, worsening illness, or are findings mixed or partially reassuring?\n"
            f"If these checks are mixed or reassuring, you should usually favor survival ('No').\n\n"
            f"--- DECISION RULE ---\n"
            f"Answer 'Yes' (patient dies in the ICU) ONLY if:\n"
            f"- There is clear, strong, and consistent evidence of severe, ongoing instability or multiorgan failure,\n"
            f"- AND the condition appears refractory to usual treatment or rapidly worsening,\n"
            f"- AND multiple modalities (when available) support a poor prognosis.\n"
            f"In all other situations (including weak, ambiguous, or single-modality risk signals), answer 'No' (patient survives).\n"
            f"- Avoid overcalling mortality: when in doubt after the consistency check, favor 'No'.\n\n"
            f"--- FINAL TASK ---\n"
            f"Based on the specialist reports, the raw patient data, the consistency check, and the decision rule above, does this patient die in the ICU?\n"
            f"Answer using exactly one word: Yes or No.\n"
            f"Answer:"
        )
    
        # Assemble in *this* order: preamble → data → final instructions
        final_prompts.append([
            {
                "role": "user",
                "content": (
                    [{"type": "text", "text": judge_preamble}]
                    + judge_content
                    + [{"type": "text", "text": judge_final_instructions}]
                )
            }
        ])


    # --- PHASE 3: Final Inference ---
    texts, probs = generate_yes_no_probability(final_prompts, model, processor, max_tokens=1)
    print("Answer:",texts)
    print("Probs:",probs)
    
    # Logging
    modality_reqs = []
    detailed_outputs = []
    
    for i in range(batch_size):
        modality_reqs.append({
            'patient_summary': 1 if 'ps' in modality_usage_tracking[i] else 0,
            'ehr_timeseries': 1 if 'ehr' in modality_usage_tracking[i] else 0,
            'radiology_report': 1 if 'rr' in modality_usage_tracking[i] else 0,
            'cxr': 1 if 'cxr' in modality_usage_tracking[i] else 0
        })
        detailed_outputs.append(f"{expert_reports[i]}\nFinal Decision: {texts[i]}")

    return _format_batch_results(batch, detailed_outputs, probs, modality_reqs)
    
def _run_agenticds_simplified_batch(batch, model, processor, args):
    """
    AgentiCDS (Gatekeeper Architecture):
    1. Gatekeeper Agent: Reviews ALL data against Rules -> Selects relevant modalities.
    2. Judge Agent: Uses the STANDARD prompt structure (same as SingleAgent), but only sees the filtered data.
    """
    # --- 0. Setup ---
    if not hasattr(args, 'rulebook_dir'):
        args.rulebook_dir = args.output_dir

    _, diagnostic_rules = _load_rulebooks(args.rulebook_dir)
    allowed_mods = _parse_allowed_modalities(args)
    max_tokens = getattr(args, 'max_new_tokens', 100)

    # --- 1. Prepare Rules for Gatekeeper ---
    def format_rule_entry(r):
        feat = r.get("Feature of Interest", "Feature")
        pat = r.get("Patterns", "")
        rule = r.get("Rule", "")
        return f"- **{feat}**: {rule} (Look for: {pat})"

    rule_context = []
    # Multimodal
    if 'multimodal' in diagnostic_rules:
        rule_context.append("### MULTIMODAL RULES ###")
        for r in diagnostic_rules['multimodal']:
            rule_context.append(format_rule_entry(r))
    # Unimodal
    for m in ['ehr', 'cxr', 'rr', 'ps']:
        if m in allowed_mods and m in diagnostic_rules:
            mod_name = m.upper()
            rule_context.append(f"\n### {mod_name} RULES ###")
            for r in diagnostic_rules[m]:
                rule_context.append(format_rule_entry(r))

    rules_str = "\n".join(rule_context)

    # --- STEP 1: The Gatekeeper (Selection Phase) ---
    print(f"   [AgentiCDS] Step 1: Selecting relevant modalities...")
    gate_prompts = []

    for p in batch:
        # Gatekeeper sees ALL available data to make the decision
        content, _ = _build_prompt_content(p, p.get('ehr_text', ''), allowed_mods, prompt_type="data_only")

        sys_header = (
            f"{STANDARD_SYS_TEXT} You are the Clinical Data Gatekeeper.\n"
            f"Review the patient data below against the Rulebook.\n\n"
            f"--- RULEBOOK ---\n"
            f"{rules_str}\n\n"
            f"--- PATIENT DATA ---\n"
        )
        
        sys_footer = (
            f"\n\n--- TASK ---\n"
            f"Which modalities contain evidence that triggers the rules above or indicates mortality risk?\n"
            f"Options: EHR (Electronic Health REcords), CXR (Chest X-ray Image), RR (Radiology Reports), PS (Patient Summary and History).\n"
            f"Select ONLY the useful ones. If a modality is normal/irrelevant, exclude it.\n\n"
            f"--- OUTPUT FORMAT ---\n"
            f"Respond strictly with the comma-separated codes.\n"
            f"Example: SELECTION: EHR, CXR, RR, PS\n"
            f"SELECTION:"
        )

        full_content = [{"type": "text", "text": sys_header}] + content + [{"type": "text", "text": sys_footer}]
        gate_prompts.append([{"role": "user", "content": full_content}])

    # Generate Selections
    gate_responses = generate_response(gate_prompts, model, processor, max_tokens=max_tokens)
    _print_debug_sample(args, batch, gate_prompts, tag="AgentiCDS Selection")
    print("Requested:",gate_responses)
    # --- STEP 2: The Judge (Standard Prediction on Filtered Data) ---
    print(f"   [AgentiCDS] Step 2: Judge Prediction on filtered data...")
    judge_prompts = []
    modality_requests_list = []
    
    for i, p in enumerate(batch):
        raw_response = gate_responses[i].upper()
        
        # 1. Parse Selection
        selected_mods = []
        if "EHR" in raw_response: selected_mods.append('ehr')
        if "CXR" in raw_response: selected_mods.append('cxr')
        if "RR" in raw_response:  selected_mods.append('rr')
        if "PS" in raw_response:  selected_mods.append('ps')
        
        # Fallback: if empty, default to PS so the model isn't blind
        if not selected_mods: selected_mods = ['ps']

        # 2. Build STANDARD Prompt
        # This function handles the "Answer: " trigger automatically when prompt_type="standard"
        content_filtered, has_image = _build_prompt_content(p, p.get('ehr_text', ''), selected_mods, prompt_type="standard")
        
        # 3. Construct Final Prompt
        # We use STANDARD_SYS_MSG to keep it identical to the baseline SingleAgent
        judge_prompts.append([{"role": "user", "content": [STANDARD_SYS_MSG] + content_filtered}])

        # Logging
        modality_requests_list.append({
            'patient_summary': 1 if 'ps' in selected_mods else 0,
            'ehr_timeseries': 1 if 'ehr' in selected_mods else 0,
            'radiology_report': 1 if 'rr' in selected_mods else 0,
            'cxr': 1 if 'cxr' in selected_mods and has_image else 0
        })

    # Generate Final Decision
    final_texts, final_probs = generate_yes_no_probability(judge_prompts, model, processor, max_tokens=1)
    print('Answer:',final_texts)
    print('Probs:',final_probs)
    # Format Output
    combined_logs = []
    for i in range(len(batch)):
        log = f"[Gatekeeper Selection]: {gate_responses[i]}\n[Judge Decision]: {final_texts[i]}"
        combined_logs.append(log)

    return _format_batch_results(batch, combined_logs, final_probs, modality_requests_list)
