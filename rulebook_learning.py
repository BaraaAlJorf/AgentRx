import os
import json
import re
from tqdm import tqdm
from agent_architectures import generate_response, _build_prompt_content, _parse_allowed_modalities

class RulebookLearner:
    """
    Diagnostic-only Rulebook Learner.

    - Starts with empty structured rulebooks for each modality.
    - Expects the LLM to return a JSON array of objects with keys:
        "Feature of Interest", "Description", "Patterns", "Rule"
    - No internal max-sample logic (sampling should be handled by the DataLoader/main).
    """

    def __init__(self, model, processor, args):
        self.model = model
        self.processor = processor
        self.args = args
        self.allowed_mods = _parse_allowed_modalities(args)

        print(f"   [Rulebook] Diagnostic-only mode. Narrative branch removed.")
        # Start with an empty rulebook for each modality
        self.diagnostic_rules = {
            'ehr': [],
            'cxr': [],
            'rr': [],
            'ps': [],
            'multimodal': []
        }

    def _audit_and_update(self, agent_name, current_rules, full_prompt_content, ground_truth, task_type):
        """
        Diagnostic-only audit, modified to only ADD new rules and never delete existing ones.

        - Requests the model to output a JSON array of rule objects with keys:
            "Feature of Interest", "Description", "Patterns", "Rule"
        - If the model outputs 'NO_CHANGE' anywhere, keep current_rules.
        - If model outputs rules, merge them with current_rules by appending only non-duplicate entries.
        """

        # Pretty-print current rules for context in the prompt
        try:
            rules_json_str = json.dumps(current_rules, indent=2)
        except Exception:
            rules_json_str = str(current_rules)

        # Convert full_prompt_content (list/dict/str) into a single string
        context_str = ""
        if isinstance(full_prompt_content, list):
            for item in full_prompt_content:
                if isinstance(item, dict) and 'text' in item:
                    context_str += item['text'] + "\n"
                else:
                    context_str += str(item) + "\n"
        else:
            context_str = str(full_prompt_content)

        print(f"\n{'='*40}")
        print(f"   [DEBUG] AUDITING AGENT: {agent_name}")
        print(f"{'='*40}")
        print(f"   [DEBUG] Full Context Input (Length: {len(context_str)} chars)")
        print(f"   [DEBUG] Ground Truth Target:\n{ground_truth}")
        print(f"   [DEBUG] Current Rules (JSON):\n{rules_json_str}")

        # Prompt instructing strict JSON output with the exact four keys
        # Note: changed wording to ask only to add or suggest new rules (no deletions).
        
                # --- multimodal / judge constraint ---
        if agent_name.lower().startswith("judge") or "judge" in agent_name.lower():

            modality_def_map = {
                "ehr": "Electronic Health Records.",
                "cxr": "Chest X-ray images.",
                "rr": "Radiology Reports.",
                "ps": "Patient Summary."
            }

            allowed_defs = []
            for m in self.allowed_mods:
                if m in modality_def_map:
                    allowed_defs.append(f"- {m}: {modality_def_map[m]}")
            allowed_modality_definitions = "\n".join(allowed_defs)
            allowed_modalities_list = ", ".join(self.allowed_mods)

            prompt = (
                "You are a MULTIMODAL JUDGE.\n"
                "Your task is to propose a small set of rules that describe RELATIONSHIPS BETWEEN DATA MODALITIES\n"
                "that, when observed together, signal the specified patient outcome.\n\n"

                "CORE CONSTRAINTS (MANDATORY):\n"
                "1) Each rule MUST be supported by evidence from AT LEAST TWO distinct allowed modalities.\n"
                "2) Rules based on a single modality alone are NOT allowed.\n"
                "3) Each rule must describe how patterns across different modalities relate to each other\n"
                "   and jointly indicate the outcome.\n"
                "4) If no valid multimodal relationship can be formed for this case, output ONLY the token: NO_CHANGE\n\n"

                "ALLOWED DATA MODALITIES FOR THIS RUN:\n"
                f"{allowed_modality_definitions}\n\n"
                f"Allowed modality abbreviations: [{allowed_modalities_list}]\n\n"

                "OUTPUT FORMAT (STRICT):\n"
                "Return a JSON ARRAY (list) of rule objects.\n"
                "Each rule object MUST contain EXACTLY the following four keys:\n"
                '  "Feature of Interest", "Description", "Patterns", "Rules"\n\n'

                "FIELD SEMANTICS:\n"
                "- \"Feature of Interest\": a short name for the MULTIMODAL relationship (not a single-modality feature).\n"
                "- \"Description\": a brief explanation of the relationship AND an explicit list of the modalities involved.\n"
                "  You MUST include the modality list using EXACTLY this format:\n"
                "      Supported modalities: [mod1, mod2]\n"
                "- \"Patterns\": a concise description of what to look for in EACH listed modality\n"
                "  and how those findings relate to each other.\n"
                "- \"Rules\": a short list of if–then style statements explaining how the multimodal pattern signals the outcome. Each rule should indicate a threshold for the outcome [low risk, moderate risk, high risk]\n\n"

                "OUTPUT RULES MUST:\n"
                "- Reference at least two modalities using the allowed abbreviations.\n"
                "- Make the cross-modal relationship explicit (not just mention modalities independently).\n"
                "- Avoid concrete examples, domain-specific numbers, or fixed thresholds unless clearly present in the data.\n"
                "- Be concise and generalizable.\n\n"

                "FORMAT REQUIREMENTS:\n"
                "- Output ONLY valid JSON (a single array).\n"
                "- Do NOT include any text outside the JSON.\n"
                "- Do NOT include explanations, commentary, or examples.\n\n"

                f"--- PATIENT DATA ---\n{context_str}\n\n"
                f"--- CASE OUTCOME: {ground_truth} ---\n\n"
                f"--- CURRENT RULES (as JSON) ---\n{rules_json_str}\n\n"
                "Output: "
            )

        else:
            prompt = (
                "Given this patient data, and the patient's outcome, please suggest additional features and rules "
                "to augment the attached rulebook for this modality.\n"
                "Do NOT remove or modify existing rules — only add new ones if they are needed to explain the outcome.\n\n"

                "--- PATIENT DATA ---\n"
                f"{context_str}\n\n"
                f"--- CASE OUTCOME: {ground_truth} ---\n\n"
                f"--- CURRENT RULES (as JSON) ---\n{rules_json_str}\n\n"

                "--- INSTRUCTIONS ---\n"
                "1) First, identify clinically meaningful *features* in this modality that help explain the outcome.\n"
                "2) For each feature, you may define ONE OR MORE rules.\n"
                "   - Multiple rules for the SAME feature are encouraged if there are different useful thresholds,\n"
                "     directions of effect, or situations (e.g. mild vs severe).\n"
                "   - When you define multiple rules for the same feature, RE-USE the exact same text in\n"
                '     the \"Feature of Interest\" field.\n\n'
                "3) If no additional rules are needed to infer the patient outcome given the attached patient data, "
                "output only the token: NO_CHANGE\n"
                "4) If additional rules would help, output a JSON array (list) of rule objects.\n"
                "   You may include multiple objects that share the same \"Feature of Interest\".\n"
                "   Each object must have exactly these keys:\n"
                '     \"Feature of Interest\", \"Description\", \"Patterns\", \"Rule\"\n'
                "   - \"Feature of Interest\": short feature name relevant to this modality (e.g., \"Hypotension\", "
                "\"Elevated lactate\").\n"
                "   - \"Description\": short description of that feature.\n"
                "   - \"Patterns\": how to detect the feature or threshold in this modality.\n"
                "   - \"Rules\": a short list of if–then style statements explaining how the data patterns signals the outcome. Each rule should indicate a threshold for the outcome [low risk, moderate risk, high risk]"
                # "(e.g., \"If feature > 4, there is a higher chance for mortality\").\n\n"
                "5) Output only valid JSON (an array). No extra commentary or surrounding text. Keep entries concise.\n"
                "6) Avoid duplicates and empty fields.\n"
                "Output: "
            )


        # Call LLM
        response = generate_response(
            [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]],
            self.model,
            self.processor,
            max_tokens=1000
        )[0]

        print(f"   [DEBUG] Raw Model Response:\n{response}")

        # Quick NO_CHANGE check
        if isinstance(response, str) and "NO_CHANGE" in response:
            print(f"   [DEBUG] Outcome: NO_CHANGE triggered.")
            return current_rules

        # Helper to normalize rule dicts for comparison
        def normalize_rule(r):
            feat = (r.get("Feature of Interest") or r.get("feature") or "").strip().lower()
            rule_text = (r.get("Rules") or r.get("rules") or "").strip().lower()
            patterns = (r.get("Patterns") or r.get("patterns") or "").strip().lower()
            # Build signature prioritizing Rule + Patterns, fallback to Feature
            sig = (rule_text, patterns, feat)
            # Normalize whitespace
            sig = tuple(" ".join(x.split()) for x in sig)
            return sig

        # Build set of signatures for existing rules
        existing_sigs = set()
        for r in current_rules:
            existing_sigs.add(normalize_rule(r))

        # First attempt: extract the first JSON array and parse it strictly
        parsed_new_rules = []
        try:
            json_text_match = re.search(r'(\[.*\])', response, flags=re.DOTALL)
            json_text = json_text_match.group(1) if json_text_match else response
            parsed = json.loads(json_text)
            if not isinstance(parsed, list):
                raise ValueError("Parsed JSON is not a list")
            for obj in parsed:
                if not isinstance(obj, dict):
                    continue
                validated_obj = {
                    "Feature of Interest": str(obj.get("Feature of Interest", obj.get("feature", "") or "")).strip(),
                    "Description": str(obj.get("Description", obj.get("description", "") or "")).strip(),
                    "Patterns": str(obj.get("Patterns", obj.get("patterns", "") or "")).strip(),
                    "Rules": str(obj.get("Rules", obj.get("rules", "") or "")).strip()
                }
                # Minimal acceptance: must have either Patterns, Feature, or Rule
                if any(validated_obj[k] for k in ["Feature of Interest", "Patterns", "Rules"]):
                    parsed_new_rules.append(validated_obj)
            if parsed_new_rules:
                print(f"   [DEBUG] Parsed {len(parsed_new_rules)} candidate new rules from JSON response.")
            else:
                print("   [DEBUG] JSON parsed but no valid new rules found — falling back to tolerant parsing.")
        except Exception as e:
            print(f"   [DEBUG] JSON parsing failed: {e}. Falling back to tolerant parsing.")

        # Fallback tolerant text parsing if strict JSON parsing yielded nothing
        if not parsed_new_rules:
            new_rules = []
            current = {}
            for line in response.splitlines():
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    # Start new rule object
                    if current:
                        new_rules.append(current)
                    current = {
                        "Feature of Interest": "",
                        "Description": "",
                        "Patterns": "",
                        "Rule": ""
                    }
                    # Remove leading number
                    body = re.sub(r'^\d+\.\s*', '', line)
                    if ':' in body:
                        k, v = body.split(':', 1)
                        k_lower = k.strip().lower()
                        if "feature" in k_lower:
                            current["Feature of Interest"] = v.strip()
                        elif "description" in k_lower:
                            current["Description"] = v.strip()
                        elif "pattern" in k_lower:
                            current["Patterns"] = v.strip()
                        elif "rule" in k_lower:
                            current["Rule"] = v.strip()
                        else:
                            current["Rule"] = body.strip()
                    else:
                        current["Rule"] = body.strip()
                elif current:
                    # Heuristic continuation: if line contains comparisons or numbers, append to Patterns
                    if "if " in line.lower() or ">" in line or "<" in line or re.search(r'\d', line):
                        current["Patterns"] += (" " + line) if current["Patterns"] else line
                    else:
                        current["Description"] += (" " + line) if current["Description"] else line

            if current:
                new_rules.append(current)

            # Clean up: trim whitespace and drop mostly-empty rules
            cleaned = []
            for r in new_rules:
                for k in r:
                    if isinstance(r[k], str):
                        r[k] = r[k].strip()
                if (r.get("Patterns") and r["Patterns"].strip()) or (r.get("Rules") and r["Rules"].strip()) or (r.get("Feature of Interest") and r["Feature of Interest"].strip()):
                    if not r.get("Feature of Interest"):
                        inferred = (r.get("Rule") or r.get("Patterns") or "").split()
                        r["Feature of Interest"] = " ".join(inferred[:4]).strip() if inferred else "Unknown feature"
                    cleaned.append(r)

            if cleaned:
                parsed_new_rules = cleaned
                print(f"   [DEBUG] Tolerant parsed {len(parsed_new_rules)} candidate new rules from text response.")
            else:
                print(f"   [DEBUG] No valid structured rules extracted by tolerant parser.")

        # If still no parsed rules, keep old rules
        if not parsed_new_rules:
            print(f"   [DEBUG] No valid structured rules extracted; keeping old rules.")
            return current_rules

        # Merge: add only rules that are not duplicates
        to_add = []
        for cand in parsed_new_rules:
            sig = normalize_rule(cand)
            if sig in existing_sigs:
                print(f"   [DEBUG] Skipping duplicate rule (Feature/Rule/Patterns signature matched): {cand.get('Rules') or cand.get('Patterns') or cand.get('Feature of Interest')}")
                continue
            # Ensure fields are present (fill missing strings)
            cand_clean = {
                "Feature of Interest": cand.get("Feature of Interest", "").strip(),
                "Description": cand.get("Description", "").strip(),
                "Patterns": cand.get("Patterns", "").strip(),
                "Rules": cand.get("Rules", "").strip()
            }
            # Re-check minimal acceptance
            if any(cand_clean[k] for k in ["Feature of Interest", "Patterns", "Rules"]):
                to_add.append(cand_clean)
                existing_sigs.add(sig)  # avoid duplicates within the same batch

        if not to_add:
            print(f"   [DEBUG] Model suggested rules but all were duplicates of existing rules. No changes made.")
            return current_rules

        merged = list(current_rules) + to_add
        print(f"   [DEBUG] Rules Updated! Existing: {len(current_rules)} | Added: {len(to_add)} | New total: {len(merged)}")
        return merged


    def train_epoch(self, train_loader):
        """
        Train epoch: iterate over train_loader and audit/update diagnostic rules.
        Note: sampling / subsampling must be handled by the DataLoader or main script.
        """
        print(f"   [Rulebook Learning] Starting Direct Audit Epoch (diagnostic only)...")

        for batch in tqdm(train_loader, desc="Auditing Rules"):
            for patient in batch:
                gt_outcome = "Death during ICU stay" if patient['labels']['in_hospital_mortality_48hr'] == 1 else "Did not Die during ICU stay"
                hid = patient.get('hadm_id')
                print(f"\n>>> Processing Patient ID: {hid} | Outcome: {gt_outcome}")

                # Audit Diagnostic Rules per modality
                for mod in ['ehr', 'cxr', 'rr', 'ps']:
                    if mod not in self.allowed_mods:
                        continue
                    mod_rules = self.diagnostic_rules.get(mod, [])

                    # Use EHR text (or modality-specific representation) as the context source
                    context_source = patient.get('ehr_text', '')

                    content, _ = _build_prompt_content(patient, context_source, [mod], prompt_type="data_only")

                    updated = self._audit_and_update(
                        f"{mod} Expert",
                        mod_rules,
                        content,
                        gt_outcome,
                        "Diagnostic"
                    )

                    # Ensure updated is a list of dicts; otherwise keep old rules
                    if isinstance(updated, list):
                        self.diagnostic_rules[mod] = updated
                    else:
                        self.diagnostic_rules[mod] = mod_rules

                # Audit Multimodal (Judge) Rules if more than one modality present
                if len(self.allowed_mods) > 1:
                    mm_rules = self.diagnostic_rules.get('multimodal', [])
                    content, _ = _build_prompt_content(patient, patient.get('ehr_text', ''), self.allowed_mods, prompt_type="data_only")

                    updated_mm = self._audit_and_update(
                        "Judge",
                        mm_rules,
                        content,
                        gt_outcome,
                        "Diagnostic"
                    )
                    if isinstance(updated_mm, list):
                        self.diagnostic_rules['multimodal'] = updated_mm

    def save_rulebooks(self, output_dir):
        print(f"   [Rulebook] Saving diagnostic rules to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "diagnostic_rules.json"), "w") as f:
            json.dump(self.diagnostic_rules, f, indent=2)
