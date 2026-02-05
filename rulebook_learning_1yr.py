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

    def _audit_and_update(self, agent_name, current_rules, full_prompt_content, ground_truth, task_type):
        """
        Directly compares Data vs. Ground Truth.
        Asks LLM: "Do current rules explain this outcome? If not, fix them."
        """
        rules_str = "\n".join(current_rules)
        
        # Parse content list to string
        context_str = ""
        if isinstance(full_prompt_content, list):
            for item in full_prompt_content:
                if isinstance(item, dict) and 'text' in item:
                    context_str += item['text'] + "\n"
        else:
            context_str = str(full_prompt_content)

        print(f"\n{'='*40}")
        print(f"   [DEBUG] AUDITING AGENT: {agent_name}")
        print(f"{'='*40}")
        print(f"   [DEBUG] Full Context Input (Length: {len(context_str)} chars):\n{context_str[:50]}")
        print(f"   [DEBUG] Ground Truth Target:\n{ground_truth[:10]}")
        print(f"   [DEBUG] Current Rules:\n{rules_str}")

        # === PROMPT: AUDIT & REPAIR ===
        if task_type == "Narrative":
            prompt = (
                f"You are a Medical Scribe Supervisor.\n"
                f"--- GOAL ---\n"
                f"Ensure the rules below force a scribe to write a summary matching the Ground Truth Ideal Summary.\n\n"
                f"--- CURRENT RULES ---\n{rules_str}\n\n"
                f"--- RAW DATA (INPUT) ---\n{context_str}\n\n"
                f"--- IDEAL SUMMARY (GROUND TRUTH) ---\n{ground_truth}\n\n"
                f"--- INSTRUCTIONS ---\n"
                f"Compare the provided patient data to the Ideal Patient Summary.\n"
                f"If current rules are sufficient to teach a scribe how to write the ideal summary based on the data, output 'NO_CHANGE'.\n"
                f"Otherwise, ADD specific rules.(e.g. 1. Make sure to summarize how the imaging tests went)\n"
                f"Output strictly the numbered list of rules or 'No_CHANGE'. "
                f"End each rule with \"\\n\" to indicate the end of the new line."
            )
        else:
            # Diagnostic (Death Prediction)
            prompt = (
                f"You are a Clinical Auditor optimizing mortality prediction rules.\n"
                f"--- CASE OUTCOME: {ground_truth} ---\n\n"
                f"--- CURRENT RULES ---\n{rules_str}\n\n"
                f"--- PATIENT DATA ---\n{context_str}\n\n"
                f"--- INSTRUCTIONS ---\n"
                f"1. Audit the case: Do the current rules correctly imply the outcome '{ground_truth}' based on the data?\n"
                f"2. If Yes, output 'NO_CHANGE'.\n"
                f"3. If No (e.g., patient died but rules don't flag their symptoms), output a REVISED rule list.\n"
                f"4. Add specific thresholds if seen. **IMPORTANT: Formulate rules to explicitly 'predict Yes' for mortality** (e.g., 'If Lactate > 4, predict Yes for Mortality').\n"
                f"5. Output strictly the numbered list of rules. \n"
                f"6. End each rule with \"\\n\" to indicate the end of the new line. \n"
                f"7. Remove any duplicate rules. \n"
            )
            

        print(f"   [DEBUG] Generated Prompt being sent to Model:\n{prompt[:30]}")

        # Generate audit
        response = generate_response([[{"role": "user", "content": [{"type": "text", "text": prompt}]}]], self.model, self.processor, max_tokens=1000)[0]

        print(f"   [DEBUG] Raw Model Response:\n{response}")

        # Check for NO_CHANGE
        if "NO_CHANGE" in response:
            print(f"   [DEBUG] Outcome: NO_CHANGE triggered.")
            return current_rules

        # Parse new rules
        new_rules = []
        for line in response.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line) and len(line.split()) > 2:
                clean_line = line.replace('**', '').replace('__', '')
                new_rules.append(clean_line)
        
        if not new_rules:
            print(f"   [DEBUG] Warning: Parsing failed or empty rules returned. Keeping old rules.")
            return current_rules
            
        print(f"   [DEBUG] Rules Updated! New count: {len(new_rules)}")
        return new_rules

    def train_epoch(self, train_loader):
        print(f"   [Rulebook Learning] Starting Direct Audit Epoch...")
        
        for batch in tqdm(train_loader, desc="Auditing Rules"):
            for patient in batch:
                gt_outcome = "Death" if patient['labels']['in_hospital_mortality_48hr'] == 1 else "Survival"
                hid = patient.get('hadm_id')
                gt_narrative = self.discharge_lookup.get(int(hid)) if hid is not None else None
                
                print(f"\n>>> Processing Patient ID: {hid} | Outcome: {gt_outcome}")

                # 1. Audit Narrative Rules (Input: EHR -> Target: GT Narrative)
                if gt_narrative:
                    content, _ = _build_prompt_content(patient, patient.get('ehr_text', ''), self.allowed_mods, prompt_type="data_only")
                    self.narrative_rules = self._audit_and_update(
                        "Narrative", 
                        self.narrative_rules, 
                        content, 
                        gt_narrative, 
                        "Narrative"
                    )

                # 2. Audit Diagnostic Rules (Input: Modality Data -> Target: GT Outcome)
                for mod in ['ehr', 'cxr', 'rr', 'ps']:
                    if mod not in self.allowed_mods: continue
                    mod_rules = self.diagnostic_rules.get(mod, [])
                    
                    # For PS/EHR, use text. For others, build specific content.
                    # We use GT Narrative as context for PS if available, otherwise EHR text.
                    context_source = gt_narrative if (mod == 'ps' and gt_narrative) else patient.get('ehr_text', '')
                    
                    content, _ = _build_prompt_content(patient, context_source, [mod], prompt_type="data_only")
                    
                    self.diagnostic_rules[mod] = self._audit_and_update(
                        f"{mod} Expert",
                        mod_rules,
                        content,
                        gt_outcome,
                        "Diagnostic"
                    )

                # 3. Audit Multimodal (Judge) Rules
                if len(self.allowed_mods) > 1:
                    mm_rules = self.diagnostic_rules.get('multimodal', [])
                    # Judge sees all modality inputs
                    content, _ = _build_prompt_content(patient, patient.get('ehr_text', ''), self.allowed_mods, prompt_type="data_only")
                    
                    self.diagnostic_rules['multimodal'] = self._audit_and_update(
                        "Judge",
                        mm_rules,
                        content,
                        gt_outcome,
                        "Diagnostic"
                    )

    def save_rulebooks(self, output_dir):
        print(f"   [Rulebook] Saving rules to {output_dir}...")
        with open(os.path.join(output_dir, "narrative_rules.txt"), "w") as f:
            f.write("\n".join(self.narrative_rules))
        with open(os.path.join(output_dir, "diagnostic_rules.json"), "w") as f:
            json.dump(self.diagnostic_rules, f, indent=4)