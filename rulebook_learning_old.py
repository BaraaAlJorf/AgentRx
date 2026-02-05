import os
import json
import re
import pandas as pd
from tqdm import tqdm
from agent_architectures import generate_response, _build_prompt_content, _parse_allowed_modalities

class RulebookLearner:
    def __init__(self, model, processor, args):
        self.model = model
        self.processor = processor
        self.args = args
        self.allowed_mods = _parse_allowed_modalities(args)
        
        # Load Ground Truth (Discharge Notes)
        print(f"   [Rulebook] Loading raw discharge notes from: {args.mimic_notes_dir}")
        discharge_csv_path = os.path.join(args.mimic_notes_dir, 'discharge.csv')
        
        if not os.path.exists(discharge_csv_path):
            raise FileNotFoundError(f"Could not find discharge.csv at {discharge_csv_path}")
            
        df_notes = pd.read_csv(discharge_csv_path, usecols=['hadm_id', 'text'])
        df_notes = df_notes.dropna(subset=['hadm_id'])
        df_notes['hadm_id'] = df_notes['hadm_id'].astype(int)
        
        self.discharge_lookup = pd.Series(df_notes.text.values, index=df_notes.hadm_id).to_dict()
        print(f"   [Rulebook] Indexed {len(self.discharge_lookup)} discharge notes.")

        self.narrative_rules = ["1. Be concise and focus on clinical relevance."]
        self.diagnostic_rules = {
            'ehr': ["1. Check for sustained tachycardia."],
            'cxr': ["1. Look for consolidation."],
            'rr': ["1. Identify acute failure keywords."],
            'ps': ["1. Note chronic comorbidities."],
            'multimodal': ["1. Prioritize physiological instability."]
        }

    def _update_rules(self, agent_name, current_rules, context, decision, ground_truth, critique_focus):
        rules_str = "\n".join(current_rules)
        
        # === BRANCH 1: NARRATIVE OPTIMIZATION ===
        if agent_name == "Narrative":
            prompt = (
                f"You are a Senior Medical Scribe Supervisor mentoring a junior scribe.\n"
                f"The junior scribe failed to include important details in a patient summary.\n\n"
                f"--- CURRENT GUIDELINES ---\n{rules_str}\n\n"
                f"--- MISSING DETAILS (CRITIQUE) ---\n{critique_focus}\n\n"
                f"--- INSTRUCTIONS ---\n"
                f"1. Based on the 'Missing Details', create specific rules to ensure this info is captured next time.\n"
                f"   (e.g., 'Always explicitly list discharge medications', 'Include the admission diagnosis').\n"
                f"2. Remove vague or redundant rules.\n"
                f"3. Output the FINAL list of rules as a clean numbered list (max 10 rules).\n"
                f"4. DO NOT include headers or reasoning. Just the numbered rules."
            )
            
        # === BRANCH 2: DIAGNOSTIC OPTIMIZATION (Experts/Judge) ===
        else:
            prompt = (
                f"You are a Senior Critical Care Attending Physician supervising a student ({agent_name}).\n"
                f"The student made a mistake in predicting 48-hour ICU mortality.\n\n"
                f"--- STUDENT'S CURRENT HEURISTICS ---\n{rules_str}\n\n"
                f"--- PATIENT CONTEXT ---\n{context}\n\n"
                f"--- MISTAKE DETAILS ---\n"
                f"Student Prediction: {decision}\n"
                f"Actual Outcome: {ground_truth}\n"
                f"Error Type: {critique_focus}\n\n"
                f"--- INSTRUCTIONS ---\n"
                f"1. Analyze why the student failed based on the clinical text provided.\n"
                f"2. Remove any bad/irrelevant rules.\n"
                f"3. Add SPECIFIC clinical rules (e.g., 'If History of Metastatic Cancer, predict Death', 'If Lactate < 2.0, predict Survival').\n"
                f"4. Output the FINAL list of rules as a clean numbered list.\n"
                f"5. DO NOT include headers, reasoning, or markdown bolding. Just the rules. Each rule must start with a number indicating its position (e.g. Rule 1.) and end with a \"\\n\"."
            )

        # === SHARED GENERATION & PARSING ===
        response = generate_response([[{"role": "user", "content": [{"type": "text", "text": prompt}]}]], self.model, self.processor, max_tokens=1000)[0]
        
        # Debugging
        print(f"\n   >>> [Rule Update: {agent_name}]")
        if agent_name == "Narrative":
            print(f"   >>> Critique used: {critique_focus}...")
        else:
            print(f"   >>> Context Snippet: {context}...")
        print(f"   >>> Raw LLM Response:\n{response}\n   >>> ---------------------")

        new_rules = []
        for line in response.split('\n'):
            line = line.strip()
            # Regex: Must start with number. 
            # For narrative, rules might be shorter (e.g. "1. List medications"), so we lower the word count check to > 2
            if re.match(r'^\d+\.', line) and len(line.split()) > 2:
                clean_line = line.replace('**', '').replace('__', '')
                new_rules.append(clean_line)
        
        if not new_rules:
            print(f"   [Warning] parsing failed for {agent_name}, keeping old rules.")
            return current_rules
            
        return new_rules

    def train_epoch(self, train_loader):
        print(f"   [Rulebook Learning] Starting Epoch...")
        for batch in tqdm(train_loader, desc="Learning Rules"):
            for patient in batch:
                gt_outcome = "Death" if patient['labels']['in_hospital_mortality_48hr'] == 1 else "Survival"
                hid = patient.get('hadm_id')
                gt_narrative = self.discharge_lookup.get(int(hid)) if hid is not None else None
                
                # 1. Train Narrative
                gen_narrative = "N/A"
                if gt_narrative:
                    sys = f"You are a Scribe. Write a narrative based on the following rules for the attached patient data.\nRules:\n" + "\n".join(self.narrative_rules)
                    content, _ = _build_prompt_content(patient, patient.get('ehr_text', ''), self.allowed_mods, prompt_type="data_only")
                    gen_narrative = generate_response([[{"role": "user", "content": [{"type": "text", "text": sys}] + content}]], self.model, self.processor, max_tokens=1000)[0]
                    
                    comp = generate_response([[{"role": "user", "content": [{"type": "text", "text": f"Generated: {gen_narrative}\nGT: {gt_narrative}\nDid I miss key details? Respond following this format: [Answer: Yes/No]. [Missing Infomration:]"}]}]], self.model, self.processor, max_tokens=5)[0]
                    print(f"comp: {comp}\n")
                    if "Yes" in comp:
                        self.narrative_rules = self._update_rules("Narrative", self.narrative_rules, f"Gen: {gen_narrative}", gen_narrative, gt_narrative, "Missing details")

                # 2. Train Experts
                expert_analyses = ""
                for mod in ['ehr', 'cxr', 'rr', 'ps']:
                    if mod not in self.allowed_mods: continue
                    mod_rules = self.diagnostic_rules.get(mod, [])
                    sys = f"You are {mod.upper()} Expert.\nRules:\n" + "\n".join(mod_rules) + f"\nNarrative: {gt_narrative}\nPredict Death (Yes/No)."
                    content, _ = _build_prompt_content(patient, patient.get('ehr_text', ''), [mod], prompt_type="data_only")
                    pred = generate_response([[{"role": "user", "content": [{"type": "text", "text": sys}] + content}]], self.model, self.processor, max_tokens=10)[0]
                    print(f"pred: {pred}\n")
                    pred_outcome = "Death" if "Yes" in pred else "Survival"
                    expert_analyses += f"[{mod}]: {pred}\n"
                    
                    if pred_outcome != gt_outcome:
                        self.diagnostic_rules[mod] = self._update_rules(f"{mod} Expert", mod_rules, content, pred_outcome, gt_outcome, "Wrong Prediction")

                # 3. Train Judge
                if len(self.allowed_mods) > 1:
                    mm_rules = self.diagnostic_rules.get('multimodal', [])
                    judge_sys = f"Judge.\nRules:\n" + "\n".join(mm_rules) + f"\nExperts:\n{expert_analyses}\nPredict Death (Yes/No)."
                    
                    content, _ = _build_prompt_content(patient, patient.get('ehr_text', ''), self.allowed_mods, prompt_type="data_only")
                    judge_pred = generate_response([[{"role": "user", "content": [{"type": "text", "text": judge_sys}] + content}]], self.model, self.processor, max_tokens=10)[0]
                    judge_outcome = "Death" if "Yes" in judge_pred else "Survival"
                    
                    # Add debug for consensus
                    print(f"   [Debug Judge] Pred: {judge_outcome} | GT: {gt_outcome} | Inputs: {expert_analyses.strip().replace(chr(10), ' ')}")
            
                    if judge_outcome != gt_outcome:
                        print(f"   [Debug Judge] FAILURE. Updating Multimodal Rules...")
                        self.diagnostic_rules['multimodal'] = self._update_rules("Judge", mm_rules, expert_analyses, judge_outcome, gt_outcome, "Wrong Consensus")
    
    def save_rulebooks(self, output_dir):
        with open(os.path.join(output_dir, "narrative_rules.txt"), "w") as f:
            f.write("\n".join(self.narrative_rules))
        with open(os.path.join(output_dir, "diagnostic_rules.json"), "w") as f:
            json.dump(self.diagnostic_rules, f, indent=4)