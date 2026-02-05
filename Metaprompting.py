import re
from agent_main import generate_response

class MedicalMetaScaffolding:
    def __init__(self, model, processor, patient_data, allowed_modalities, max_tokens=256):
        self.model = model
        self.processor = processor
        self.patient_data = patient_data
        self.allowed_modalities = allowed_modalities
        self.max_tokens = max_tokens
        
        # Build clearer tool descriptions
        self.tool_descriptions = []
        if 'cxr' in allowed_modalities: 
            self.tool_descriptions.append("- Expert CXR: Can view the actual Chest X-Ray image.")
        if 'ehr' in allowed_modalities: 
            self.tool_descriptions.append("- Expert EHR: Can analyze time-series vitals and labs.")
        if 'rr' in allowed_modalities: 
            self.tool_descriptions.append("- Expert RR: Can read the full radiology text reports.")

    def generate_expert_response(self, expert_name, instruction):
        """Executes the specific expert logic."""
        if expert_name == "Expert CXR":
            has_image = False
            img_input = None
            if 'pil_image' in self.patient_data and self.patient_data['pil_image']:
                 img_input = self.patient_data['pil_image']
                 has_image = True
            elif self.patient_data.get('cxr_image_path') != 'CXR not available':
                try: 
                    img_input = Image.open(self.patient_data['cxr_image_path']).convert("RGB")
                    has_image = True
                except: pass
            
            if not has_image: return "System: CXR image not found."

            content = [
                {"type": "image", "image": img_input},
                {"type": "text", "text": f"You are a Radiologist. {instruction}"}
            ]
            return generate_response([[{"role": "user", "content": content}]], self.model, self.processor, self.max_tokens)[0]

        elif expert_name == "Expert EHR":
            ehr_text = self.patient_data.get('ehr_text', "No EHR data.")
            content = [{"type": "text", "text": f"You are an ICU Data Analyst.\nData:\n{ehr_text}\n\nTask: {instruction}"}]
            return generate_response([[{"role": "user", "content": content}]], self.model, self.processor, self.max_tokens)[0]

        elif expert_name == "Expert RR":
            report = self.patient_data.get('radiology_report_text', "No report.")
            content = [{"type": "text", "text": f"You are a Text Specialist.\nReport:\n{report}\n\nTask: {instruction}"}]
            return generate_response([[{"role": "user", "content": content}]], self.model, self.processor, self.max_tokens)[0]
            
        return "System: Expert name not recognized."

    def run_meta_loop(self):
        """
        Runs the reasoning loop. 
        Returns history formatted for the final probability probe.
        """
        ps = self.patient_data.get('patient_summary_text', '')
        
        # --- ENHANCED SYSTEM PROMPT ---
        tools_block = "\n".join(self.tool_descriptions)
        
        system_text = (
            f"You are the Lead ICU Physician. Your goal is to determine if the patient will die in the ICU.\n\n"
            f"--- PATIENT SUMMARY ---\n{ps}\n\n"
            f"--- AVAILABLE TOOLS ---\n"
            f"You have a team of experts. You MUST consult them if the summary is insufficient.\n"
            f"{tools_block}\n\n"
            f"--- INSTRUCTIONS ---\n"
            f"1. CONSULT: To use a tool, output the name and the question inside triple quotes.\n"
            f"   Example:\n"
            f"   Expert CXR:\n"
            f"   \"\"\"\n"
            f"   Is there consolidation or pleural effusion?\n"
            f"   \"\"\"\n\n"
            f"2. DECIDE: When you have enough information, output the final answer exactly as:\n"
            f"   Answer: Yes (or Answer: No)\n"
        )
        
        history = [{"role": "user", "content": [{"type": "text", "text": system_text}]}]

        for _ in range(5): 
            # Generate thought
            response = generate_response([history], self.model, self.processor, max_tokens=150)[0]
            
            # Check for Final Answer pattern matching the instructions
            if "Answer: Yes" in response or "Answer: No" in response:
                # Clean up the response to end exactly at "Answer: "
                # This prepares the context for the probability check.
                clean_response = response.split("Answer:")[0].strip() + "\nAnswer: "
                history.append({"role": "assistant", "content": [{"type": "text", "text": clean_response}]})
                return history

            # Append assistant thought
            history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

            # Check for Expert Calls
            matches = re.findall(r"(Expert \w+(?: \w+)?):\n\"\"\"(.*?)\"\"\"", response, re.DOTALL)
            
            if matches:
                obs = ""
                for name, instr in matches:
                    result = self.generate_expert_response(name, instr.strip())
                    obs += f"[{name} Output]: {result}\n\n"
                history.append({"role": "user", "content": [{"type": "text", "text": obs}]})
            else:
                # If no tool called and no answer, nudge the model
                history.append({"role": "user", "content": [{"type": "text", "text": "Proceed. Use a tool or provide 'Answer:'."}]})

        # Timeout Fallback: Append the standard prompt to force probability extraction
        fallback = {
            "role": "user", 
            "content": [{"type": "text", "text": "\nDoes this patient die in the ICU? Answer only using one word - Yes or No\nAnswer: "}]
        }
        history.append(fallback)
        return history