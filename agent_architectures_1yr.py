import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, GenerationConfig, AutoConfig
from transformers import PaliGemmaForConditionalGeneration as AutoModelForMultimodalLM
from transformers.generation import GenerationMixin
from PIL import Image
from datasets.data_utils import load_few_shot_data
import re
import json

# --------------------------------------------------------------------
# 0. Class definition
# --------------------------------------------------------------------

# --- UNIFIED PROMPT HEADERS ---
# The raw text for f-strings
STANDARD_SYS_TEXT = "You are an expert ICU risk prediction model. This patient was just discharged."

# The dictionary object for content lists
STANDARD_SYS_MSG = {"type": "text", "text": STANDARD_SYS_TEXT + "\n\n"}

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
                {"type": "text", "text": f"You are a Chest X-ray Specialist. The patient of interest's chest x-ray is attached. {instruction}"}
            ]
            return generate_response([[{"role": "user", "content": content}]], self.model, self.processor, self.max_tokens)[0]

        elif expert_name == "Expert EHR":
            ehr_text = self.patient_data.get('ehr_text', "No EHR data.")
            content = [{"type": "text", "text": f"You are an Electronic Health Record Specialist.\n The patient of interest's data:\n{ehr_text}\n\nTask: {instruction}"}]
            return generate_response([[{"role": "user", "content": content}]], self.model, self.processor, self.max_tokens)[0]

        elif expert_name == "Expert RR":
            report = self.patient_data.get('radiology_report_text', "No report.")
            content = [{"type": "text", "text": f"You are a Radiology Report Specialist.\n The patient of interest's report:\n{report}\n\nTask: {instruction}"}]
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
            f"{STANDARD_SYS_TEXT} Your goal is to determine if the patient will die in one year post discharge.\n\n"
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
            "content": [{"type": "text", "text": "\nDoes this patient die within one year post discharge from the ICU? Answer only using one word - Yes or No\nAnswer: "}]
        }
        history.append(fallback)
        return history


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
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
        if "medgemma" in model_id.lower():
            processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
            model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
    
        elif "Qwen" in model_id:
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
            
            try:
                if hasattr(model, 'language_model'):
                    lm_model = model.language_model
                    lm_class = lm_model.__class__
                    
                    # 1. Patch GenerationMixin (Fixes missing .generate())
                    if not hasattr(lm_class, 'generate') or GenerationMixin not in lm_class.__bases__:
                        print(f"🔧 [Fix] Patching {lm_class.__name__} with GenerationMixin...")
                        lm_class.__bases__ = (GenerationMixin,) + lm_class.__bases__
                    
                    # 2. Patch GenerationConfig (Fixes 'NoneType' has no attribute '_from_model_config')
                    if not hasattr(lm_model, 'generation_config') or lm_model.generation_config is None:
                        print(f"🔧 [Fix] Creating missing GenerationConfig for {lm_class.__name__}...")
                        lm_model.generation_config = GenerationConfig.from_model_config(lm_model.config)
            
            except Exception as patch_e:
                print(f"⚠️ Warning: Failed to patch InternVL compatibility: {patch_e}")# ------------------------------------------------
            
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            if tokenizer:
                tokenizer.padding_side = "left"
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                img_context_token_id = tokenizer.convert_tokens_to_ids('<img_context>')
                model.img_context_token_id = img_context_token_id
                print(f"✅ [Fix Applied] Set InternVL img_context_token_id to: {img_context_token_id}")
            else:
                print("⚠️ Warning: Could not find tokenizer to set img_context_token_id")
        elif "Phi-4" in model_id:
            print(f"Loading {model_id} with Forced Eager Attention...")
            
            # 1. Manually load the config
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
            
            # 2. Force-overwrite the attention implementation
            config.attn_implementation = "eager"
            
            # 3. Load the model using this sanitized config
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                config=config, 
                **load_kwargs
            )
        elif "Ovis" in model_id:
            
            ovis_kwargs = load_kwargs.copy()
            ovis_kwargs.pop("device_map", None) 
            
            model = AutoModelForCausalLM.from_pretrained(model_id, **ovis_kwargs).to("cuda")
        elif "llava" in model_id.lower():
            print(f"Loading {model_id} as LlavaForConditionalGeneration...")
            from transformers import LlavaForConditionalGeneration
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=load_kwargs.get("torch_dtype", None),
                device_map=load_kwargs.get("device_map", None),
                low_cpu_mem_usage=load_kwargs.get("low_cpu_mem_usage", None),
                token=hf_token
            )

        else:
            print(f"Loading {model_id} with generic AutoModel fallback...")
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            except:
                model = AutoModel.from_pretrained(model_id, **load_kwargs)

        model.eval()
        # make model_id available downstream
        try:
            model.model_id = model_id
        except Exception:
            pass

        
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
        
def _prepare_inputs_for_vlm(prompts, processor, device, model_id):
    """
    Robustly prepares inputs. 
    - Qwen: Uses native qwen_vl_utils.
    - Ovis: Preserves batch alignment (List of Tensors).
    - InternVL/Generic: Flattens all images into a single stack (Tensor + Flags).
    """
    # --- 0. Detect Model Type ---
    # We check the processor class name to determine the backbone
    mid = (model_id or "").lower()
    is_qwen = "qwen" in mid
    is_ovis = "ovis" in mid
    is_medgemma = "medgemma" in mid
    is_huatuo = "huatuo" in mid
    is_llava = "llava" in mid
    
    # --- 1. Qwen Handling (Native support) ---
    # --- 1. MedGemma / PaliGemma Handling --
    if is_medgemma:
        # ---- MedGemma / PaliGemma handling (fixed) ----
        # The Gemma processor expects pre-rendered prompt strings (it searches for boi_token).
        print("MEDGEMMA handler (fixed): producing prompt strings for processor")

        prompt_texts = []
        images_grouped = []

        for conversation in prompts:
            # conversation is expected: [{"role":"user","content":[...]}] or similar
            try:
                # Use the processor's chat template to produce the textual prompt.
                # We keep tokenize=False so we get the raw string; the processor will then
                # search this string for its image placeholder tokens.
                rendered = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            except Exception as e:
                # Fallback: attempt to stringify the conversation safely
                print(f"⚠️ medgemma: apply_chat_template failed: {e}; falling back to str()")
                rendered = str(conversation)

            # Ensure we have a pure Python str here
            if not isinstance(rendered, str):
                print(f"⚠️ medgemma: rendered prompt is {type(rendered)}; converting to str()")
                rendered = str(rendered)

            # Remove any accidental leading <bos> token if processor adds it
            try:
                rendered = rendered.replace("<bos>", "")
            except Exception:
                pass

            prompt_texts.append(rendered)

            # Collect images for this conversation (list or None)
            conv_images = []
            for msg in conversation:
                if isinstance(msg.get('content'), list):
                    for item in msg['content']:
                        if item.get('type') == 'image':
                            conv_images.append(item['image'])
            images_grouped.append(conv_images if conv_images else None)

        # Optional debug: show types
        for i, txt in enumerate(prompt_texts):
            if not isinstance(txt, str):
                print(f"⚠️ medgemma debug: prompt_texts[{i}] is {type(txt)}")

        # Now call the processor with strings + grouped images
        inputs = processor(
            text=prompt_texts,
            images=images_grouped if any(imgs is not None for imgs in images_grouped) else None,
            return_tensors="pt",
            padding=True,
        ).to(device)

        return inputs
    elif is_llava:
        # LLAVA / llava-med handler (follows the quick-start example)
        print("LLAVA handler: using tokenizer.apply_chat_template + processor(images=..., text=...)")

        prompt_texts = []
        images_grouped = []  # list per-batch-item of either a list of PIL images or None

        for conversation in prompts:
            # conversation is a list of message dicts, e.g. [{"role":"user","content":[...]}]
            # We want to render a single prompt string using the tokenizer's chat template.
            try:
                # Prefer tokenizer.apply_chat_template as in LLava quick-start
                prompt = processor.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                # Fallback: if processor provides an apply_chat_template, try that,
                # otherwise fall back to a simple string representation to avoid crashing.
                try:
                    if hasattr(processor, "apply_chat_template"):
                        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                    else:
                        prompt = str(conversation)
                except Exception:
                    prompt = str(conversation)

            prompt_texts.append(prompt)

            # Collect PIL images (keep per-conversation list). The processor expects
            # the `images` argument to align with the prompts.
            conv_images = []
            for msg in conversation:
                # many of your messages use 'content' as a list
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            # item['image'] should be a PIL.Image if available
                            img_obj = item.get("image", None)
                            if img_obj is not None:
                                conv_images.append(img_obj)
            images_grouped.append(conv_images if conv_images else None)

        # If *any* sample has images, pass the images list to the processor; else omit it.
        images_arg = images_grouped if any(imgs is not None for imgs in images_grouped) else None

        # Call processor exactly like Quick Start: text=prompt(s), images=...
        inputs = processor(
            text=prompt_texts,
            images=images_arg,
            return_tensors="pt",
            padding=True,
        ).to(device)

        return inputs





    # # --- 2. Huatuo Handling ---
    # elif is_huatuo:
    #     formatted_texts = []
    #     all_images = []
    #     for conversation in prompts:
    #         full_text = ""
    #         for msg in conversation:
    #             text_part = ""
    #             for item in msg['content']:
    #                 if item['type'] == 'text':
    #                     text_part += item['text']
    #                 elif item['type'] == 'image':
    #                     # Huatuo usually requires the <image> tag in text
    #                     if "<image>" not in text_part:
    #                         text_part = "<image>\n" + text_part
    #                     all_images.append(item['image'])
                
    #             # Manual template for Huatuo if apply_chat_template is finicky
    #             full_text += f"User: {text_part}\nAssistant: "
    #         formatted_texts.append(full_text)

    #     inputs = processor(text=formatted_texts, images=all_images, return_tensors="pt", padding=True).to(device)
    #     return inputs
    elif is_qwen:
        try:
            print("adapting input to qwen")
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
            pass

    # --- 2. Ovis Handling (Preserve Batch Alignment) ---
    elif is_ovis:
        print("adapting input to ovis")
        formatted_texts = []
        
        # Initialize pixel_values as a LIST of None, size = Batch Size
        batch_pixel_values = [None] * len(prompts) 
        
        has_images = False
    
        for i, conversation in enumerate(prompts):
            flattened_conv = []
            patient_images = [] # Collect images ONLY for this patient
            
            for msg in conversation:
                if isinstance(msg['content'], list):
                    text_part = ""
                    img_count = 0
                    for item in msg['content']:
                        if item['type'] == 'text':
                            text_part += item['text']
                        elif item['type'] == 'image':
                            patient_images.append(item['image'])
                            img_count += 1
                            has_images = True
                        
                    # Add <image> token if needed (Ovis usually handles this via processor, but good to ensure)
                    if img_count > 0 and "<image>" not in text_part:
                        text_part = ("<image>\n" * img_count) + text_part
                    
                    flattened_conv.append({"role": msg['role'], "content": text_part})
                else:
                    flattened_conv.append(msg)
    
            # Apply Chat Template
            if hasattr(processor, "tokenizer"):
                txt = processor.tokenizer.apply_chat_template(flattened_conv, tokenize=False, add_generation_prompt=True)
            else:
                txt = processor.apply_chat_template(flattened_conv, tokenize=False, add_generation_prompt=True)
            formatted_texts.append(txt)
    
            # Process images JUST for this patient
            if patient_images:
                # Returns Dict with 'pixel_values' -> Tensor
                processed = processor.image_processor(images=patient_images, return_tensors="pt")
                
                # Ovis expects a LIST of Tensors (one per batch item)
                batch_pixel_values[i] = processed.pixel_values.to(device)
    
        # Tokenize Text
        inputs = processor(
            text=formatted_texts,
            return_tensors="pt",
            padding=True
        ).to(device)
    
        # Attach the aligned list of pixel values
        inputs["pixel_values"] = batch_pixel_values
        return inputs

    # --- 3. InternVL / Phi-4 / Generic Handling (Flatten & Stack) ---
    else:
        print("GENERAL GENERAL GENERALLLL")
        formatted_texts = []
        all_images = []
        has_images = False
    
        for conversation in prompts:
            flattened_conv = []
            for msg in conversation:
                if isinstance(msg['content'], list):
                    # --- FLATTENING LOGIC ---
                    text_part = ""
                    img_count = 0
                    for item in msg['content']:
                        if item['type'] == 'text':
                            text_part += item['text']
                        elif item['type'] == 'image':
                            all_images.append(item['image'])
                            img_count += 1
                            has_images = True
                        
                    # Add <image> token for InternVL if images exist
                    if img_count > 0 and "<image>" not in text_part:
                        text_part = ("<image>\n" * img_count) + text_part
                    
                    flattened_conv.append({"role": msg['role'], "content": text_part})
                else:
                    flattened_conv.append(msg)
    
            # Apply template
            if hasattr(processor, "tokenizer"):
                txt = processor.tokenizer.apply_chat_template(flattened_conv, tokenize=False, add_generation_prompt=True)
            else:
                txt = processor.apply_chat_template(flattened_conv, tokenize=False, add_generation_prompt=True)
            
            formatted_texts.append(txt)
            
        # 1. Process Text Only
        inputs = processor(
            text=formatted_texts,
            return_tensors="pt",
            padding=True
        )
        
        # 2. Process Images Separately & Merge
        if has_images and hasattr(processor, "image_processor"):
            image_inputs = processor.image_processor(images=all_images, return_tensors="pt")
            inputs["pixel_values"] = image_inputs.pixel_values
            
        return inputs.to(device)
        
def _get_dummy_pixel_values(batch_size, device, dtype):
    """Creates a black dummy image tensor for InternVL to prevent crashes on text-only batches."""
    # Shape: [Batch, Num_Images=1, Channels=3, H=448, W=448]
    return torch.zeros((batch_size, 3, 448, 448), device=device, dtype=dtype)

def generate_response(prompts, model, processor, max_tokens, **generation_kwargs):
    model_id = getattr(model, "model_id", "") or ""
    inputs = _prepare_inputs_for_vlm(prompts, processor, model.device, model_id=getattr(model, "model_id", None))
    
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw", None)
    
    model_name = model.__class__.__name__

    try:
        with torch.inference_mode():
            # BRANCH 1: Ovis (Requires inputs=... and pixel_values=[])
            if "medgemma" in model_id:
                # MedGemma returns full sequence; we must slice out the input
                outputs = model.generate(**inputs, max_new_tokens=max_tokens, **generation_kwargs)
                input_len = inputs["input_ids"].shape[-1]
                generated_ids = outputs[0][input_len:]
                return [processor.decode(generated_ids, skip_special_tokens=True)]
            elif "Ovis" in model_name:
                gen_args = {
                    "inputs": input_ids,        # FIX: 'inputs' -> 'input_ids'
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_tokens, # FIX: Silence warning
                    **generation_kwargs
                }
                
                # FIX: Ovis Pixel Value Handling (List of Tensors or None)
                if pixel_values is None:
                    batch_size = input_ids.shape[0]
                    gen_args["pixel_values"] = [None] * batch_size 
                else:
                    # pixel_values comes from _prepare_inputs as a List[Tensor | None]
                    gen_args["pixel_values"] = pixel_values
                
                outputs = model.generate(**gen_args)
            
            # BRANCH 2: InternVL / Qwen / Standard Models
            else:
                # --- FIX FOR INTERNVL EMPTY OUTPUTS ---
                gen_args = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_tokens,
                    **generation_kwargs
                }

                if "Intern" in model_name:
                    if pixel_values is None:
                        # CRITICAL FIX: Do NOT create dummy tensors. 
                        # Pass None explicitly to tell InternVL this is text-only.
                        gen_args["pixel_values"] = None
                        
                        # Also do NOT pass image_flags if pixel_values is None
                        if "image_flags" in gen_args:
                            del gen_args["image_flags"]
                    else:
                        # Only pass pixel_values and flags if actual images exist
                        gen_args["pixel_values"] = pixel_values
                        # Ensure flags are present if not already in input
                        if "image_flags" not in inputs:
                             # If processor didn't create flags, create default ones (1s)
                             # assuming the pixel_values correspond to real images.
                             # (Usually the processor handles this, but strictly speaking:)
                             gen_args["image_flags"] = torch.ones((input_ids.shape[0], 1), device=model.device, dtype=torch.long)
                        else:
                             gen_args["image_flags"] = inputs["image_flags"]

                elif pixel_values is not None:
                    # For Qwen etc, only pass if present
                    gen_args["pixel_values"] = pixel_values
                    if image_grid_thw is not None:
                        gen_args["image_grid_thw"] = image_grid_thw
                
                outputs = model.generate(**gen_args)

    except RuntimeError as e:
        if "out of memory" in str(e).lower(): return ["Error: OOM"] * len(prompts)
        raise
        
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)

def generate_yes_no_probability(prompts, model, processor, max_tokens=1):
    model_id = getattr(model, "model_id", "") or ""
    inputs = _prepare_inputs_for_vlm(prompts, processor, model.device, model_id=getattr(model, "model_id", None))
    
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values", None)
    
    model_name = model.__class__.__name__

    try:
        with torch.inference_mode():
            # BRANCH 1: Ovis
            if "Ovis" in model_name:
                forward_args = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": None
                }
                # Apply the same Empty List workaround
                if pixel_values is None:
                    batch_size = input_ids.shape[0]
                    forward_args["pixel_values"] = [None] * batch_size
                else:
                    forward_args["pixel_values"] = pixel_values
                
                outputs = model(**forward_args)
            
            # BRANCH 2: InternVL / Qwen / Standard Models
            else:
                if "Intern" in model_name:
                    if pixel_values is None:
                        # Create dummy black image to prevent 'NoneType' squeeze error
                        pixel_values = _get_dummy_pixel_values(input_ids.shape[0], model.device, model.dtype)
                        inputs["image_flags"] = torch.zeros((input_ids.shape[0], 1), device=model.device, dtype=torch.long)
                    # InternVL requires pixel_values to be passed explicitly
                    inputs["pixel_values"] = pixel_values
                outputs = model(**inputs)

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

    # Decode top token
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
                '1yr_mortality': 1 if exp_prob > 0.5 else 0, 
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
    if prompt_type == "data_only":
        prompt_text = data_block
    elif outcome is not None:
        # Few-Shot Example
        outcome_str = "Yes" if outcome == 1 else "No"
        prompt_text = (
            f"{data_block}\n\n"
            "--- DECISION ---\n"
            "Does this patient die within one year post discharge from the ICU? Answer only using one word - Yes or No?\n"
            f"Answer: {outcome_str}\n\n"
            "--------------------------------------------------\n"
        )
    elif prompt_type == "cot_reasoning":
        # CoT Step 1
        prompt_text = (
            f"{data_block}\n\n"
            "--- ANALYSIS ---\n"
            "Analyze the patient's condition step by step. Consider vitals, labs, and history. "
            "Identify key risk factors for imminent mortality within one year post discharge from the ICU. Keep the reasoning within 2-4 sentences.\n"
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
            "Provide constructive feedback to improve the one-year mortality assessment. "
            "Do NOT output a final decision yet, just the feedback. Keep the feedback within 2-4 sentences.\n"
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
            "Based on the analysis above, does this patient die within one year post discharge from the ICU? Answer only using one word - Yes or No\n"
            "Answer: "
        )
    else:
        # Standard / Single Agent
        prompt_text = (
            f"{data_block}\n\n"
            "--- DECISION ---\n"
            "Does this patient die within one year post discharge from the ICU? Answer only using one word - Yes or No\n"
            "Answer: " 
        )

    content.append({"type": "text", "text": prompt_text})
    return content, has_image

def _parse_generated_confidence(text, default=50.0):
    """
    Extracts 'Confidence: X%' from ReConcile outputs.
    Robust to Markdown bolding (e.g., **Confidence**: 80%).
    """
    if not text: return default / 100.0
    
    # Remove asterisks used for bolding (Common in Qwen/InternVL)
    clean_text = text.replace('*', '')

    # Regex handles: "Confidence: 80", "Certainty: 0.9", "Score: 80%"
    match = re.search(r"(?:Confidence|Certainty|Score)\s*[:=]\s*(\d+(?:\.\d+)?)", clean_text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        # Normalize 0-100 to 0-1
        if val > 1.0: val = val / 100.0
        return min(max(val, 0.0), 1.0)
    
    return default / 100.0

def _chunk_ehr_sequence(ehr_text, num_chunks=4):
    """
    Splits EHR text into 'num_chunks' sequential blocks based on line count.
    Best for irregular time-series where specific time markers might be sparse.
    """
    if not ehr_text: return ["No Data"] * num_chunks
    
    lines = ehr_text.strip().split('\n')
    total_lines = len(lines)
    
    # If very short, just return the whole thing in the first chunk
    if total_lines < num_chunks:
        return [ehr_text] + ["(No further data)"] * (num_chunks - 1)
        
    chunk_size = np.ceil(total_lines / num_chunks).astype(int)
    chunks = []
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_lines)
        
        if start >= total_lines:
            chunks.append("(No further data)")
        else:
            block = "\n".join(lines[start:end])
            chunks.append(block)
            
    return chunks
    
def _load_rulebooks(rulebook_dir):
    narrative_rules = ["1. Summarize clinical data."] 
    diagnostic_rules = {}
    
    n_path = os.path.join(rulebook_dir, "narrative_rules.txt")
    d_path = os.path.join(rulebook_dir, "diagnostic_rules.json")
    
    if os.path.exists(n_path):
        with open(n_path, 'r') as f: narrative_rules = [l.strip() for l in f.readlines()]
    if os.path.exists(d_path):
        with open(d_path, 'r') as f: diagnostic_rules = json.load(f)
        
    return narrative_rules, diagnostic_rules


# --------------------------------------------------------------------
# 3. Agent Implementations
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
        system_msg = {"type": "text", "text": "You are an expert ICU risk prediction model. This patient was just discharged.\n\n"}
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
            shot_outcome = shot_patient['labels']['1yr_mortality']
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
        reasoning_prompts.append([{"role": "user", "content": [STANDARD_SYS_MSG] + content}])
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
            reasoning_prompts.append([{"role": "user", "content": [STANDARD_SYS_MSG] + content}])
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
        prompts.append([{"role": "user", "content": [STANDARD_SYS_MSG] + content}])
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

    configs = [('ps', 'PS', "Predict mortality within one year based ONLY on the summary."), 
               ('ehr', 'EHR', "Predict mortality within one year based ONLY on the vitals."), 
               ('rr', 'RR', "Predict mortality within one year based ONLY on the radiology report."), 
               ('cxr', 'CXR', "Predict mortality within one year based ONLY on the chest X-ray.")]
    
    for mod_key, mod_name, sys_desc in configs:
        if mod_key in allowed_mods:
            prompts = []
            for p in batch:
                content, _ = _build_prompt_content(p, p.get('ehr_text', ""), [mod_key], prompt_type="standard")
                prompts.append([{"role": "user", "content": [{"type": "text", "text": f"{STANDARD_SYS_TEXT}\n{sys_desc}\n\n"}] + content}])
            _, probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
            all_voter_probs.append(probs)
            active_voters.append(mod_name)

    if not all_voter_probs: return _format_batch_results(batch, ["No Data"]*batch_size, np.zeros(batch_size), [])
    avg_probs = np.mean(all_voter_probs, axis=0)
    final_texts = [f"Vote UniModal ({'+'.join(active_voters)})"] * batch_size
    return _format_batch_results(batch, final_texts, avg_probs, [{'patient_summary': 1, 'ehr_timeseries': 1, 'radiology_report': 1, 'cxr': 1}] * batch_size)

def _run_debate_unimodal_batch(batch, model, processor, args):
    """
    Consensus Debate (UniModal) - 'Answer First' Variation.
    Flow:
    1. Probe 'Answer:' for probabilities -> Check Consensus.
    2. If Consensus -> Stop & Average.
    3. If No Consensus -> Generate 'Reasoning' -> Pass to Peers -> Repeat.a
    """
    allowed_mods = _parse_allowed_modalities(args)
    max_tokens = 100  # Limited to keep reasoning concise (3-5 sentences)
    n_rounds = getattr(args, 'debate_rounds', 3)
    
    # 1. Define Potential Agents
    potential_agents = [
        ('ehr', 'EHR Specialist', ['ehr'], "You are an Electronic Health Record Specialist. This patient was just discharged. Analyze the vitals and labs."),
        ('cxr', 'CXR Specialist', ['cxr'], "You are a Chest X-ray Specialist. This patient was just discharged. Analyze the image."),
        ('ps', 'Patient Summary Specialist', ['ps'], "You are a Clinical Historian. This patient was just discharged. Analyze the patient summary."),
        ('rr', 'Radiology Report Specialist', ['rr'], "You are a Radiology Report Specialist. This patient was just discharged. Analyze the text report.")
    ]
    
    # 2. Select Active Agents
    agents = []
    for mod_key, name, mod_list, sys_prompt in potential_agents:
        if mod_key in allowed_mods:
            agents.append({'name': name, 'mods': mod_list, 'sys': sys_prompt})
            
    if not agents:
        return _format_batch_results(batch, ["No Agents"]*len(batch), np.zeros(len(batch)), [])

    # State: [Agent_Index][Batch_Index]
    current_reasoning = [["No prior reasoning."] * len(batch) for _ in range(len(agents))]
    batch_indices = list(range(len(batch))) # Track who is still debating
    final_probs = np.zeros(len(batch))
    final_texts = [""] * len(batch)
    
    # Track final modality usage for reporting
    modality_requests = [{'patient_summary': 1, 'ehr_timeseries': 1, 'radiology_report': 1, 'cxr': 1}] * len(batch)

    print(f"   [Debate UniModal] Starting debate with {len(agents)} agents for {n_rounds} rounds...")

    for r in range(n_rounds):
        # If everyone has reached consensus, break early
        if not batch_indices:
            break

        print(f"   [Round {r+1}] Processing {len(batch_indices)} active debates...")
        
        # A. Collect votes from ALL agents for active patients
        round_probs = [] # [Agent][Active_Index]
        round_decisions = [] # [Agent][Active_Index] (Text "Yes"/"No")
        
        for k, agent in enumerate(agents):
            prompts = []
            for i in batch_indices:
                # Build Peer Context
                peer_txt = ""
                if r > 0:
                    peer_txt = "\n--- COLLEAGUE OPINIONS (Previous Round) ---\n"
                    for j, peer in enumerate(agents):
                        if k != j: 
                            peer_txt += f"{peer['name']}: {current_reasoning[j][i]}\n"
                
                # Build Input Content
                content, _ = _build_prompt_content(batch[i], batch[i].get('ehr_text', ""), agent['mods'], prompt_type="data_only")
                
                # Strict Format Prompt
                txt = (
                    f"{agent['sys']}\n"
                    f"--- TASK ---\n"
                    f"Does this patient die within one year post discharge from the ICU?\n"
                    f"Below are your peers' opinions.\n"
                    f"{peer_txt}\n"
                    f"Based on the attached data and your peers' opinions, provide your answer and a concise explanation (3-5 sentences).\n"
                    f"Format: 'Answer: [Yes/No] because [Reasoning]'\n\n"
                    f"Answer:" # Ends exactly here for probing
                )
                prompts.append([{"role": "user", "content": [{"type": "text", "text": txt}] + content}])
            
            # --- DEBUG INSERTION ---
            # Print the prompt before probing to see exactly what the agent sees
            _print_debug_sample(args, batch, prompts, tag=f"Debate UniModal Round {r+1} - {agent['name']}")
            
            # Step 1: MEASURE PROBABILITY (Consensus Check)
            decisions, probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
            round_probs.append(probs)
            round_decisions.append(decisions)

        # B. Check Consensus per patient
        new_batch_indices = []
        
        # Transpose to iterate by patient: [Patient][Agent]
        round_probs_T = np.array(round_probs).T 
        
        for idx_in_active, global_idx in enumerate(batch_indices):
            p_vals = round_probs_T[idx_in_active]
            
            # Consensus Definition: All > 0.5 OR All <= 0.5
            all_yes = np.all(p_vals > 0.5)
            all_no = np.all(p_vals <= 0.5)
            
            is_consensus = all_yes or all_no
            is_last_round = (r == n_rounds - 1)

            if is_consensus or is_last_round:
                # Consensus Reached or Forced Stop -> Average Probabilities
                avg_prob = np.mean(p_vals)
                final_probs[global_idx] = avg_prob
                
                # Format final text log
                status = "CONSENSUS" if is_consensus else "MAX_ROUNDS"
                log = f"[{status} Round {r+1}]\n"
                for k, agent in enumerate(agents):
                    log += f"{agent['name']}: {round_decisions[k][idx_in_active]} ({p_vals[k]:.2f})\n"
                final_texts[global_idx] = log
            else:
                # No Consensus -> Keep for next round
                new_batch_indices.append(global_idx)

        # C. Generate Reasoning ONLY for those continuing
        if new_batch_indices:
            # Map global index back to the index in the CURRENT round lists
            # We need to regenerate the inputs for the generation phase
            indices_map = {g_idx: i for i, g_idx in enumerate(batch_indices)}
            
            for k, agent in enumerate(agents):
                gen_prompts = []
                # Re-construct prompts only for continuing patients
                for global_idx in new_batch_indices:
                    idx_in_round = indices_map[global_idx]
                    
                    # Same logic as above to rebuild context
                    peer_txt = ""
                    if r > 0:
                        peer_txt = "\n--- COLLEAGUE OPINIONS ---\n"
                        for j, peer in enumerate(agents):
                            if k != j: peer_txt += f"{peer['name']}: {current_reasoning[j][global_idx]}\n"
                    
                    content, _ = _build_prompt_content(batch[global_idx], batch[global_idx].get('ehr_text', ""), agent['mods'], prompt_type="data_only")
                    
                    # Prompt for GENERATION (continuing from 'Answer:')
                    txt = (
                        f"{agent['sys']}\n"
                        f"--- TASK ---\n"
                        f"Does this patient die within one year post discharge from the ICU?\n"
                        f"Below are your peers' opinions.\n"
                        f"{peer_txt}\n"
                        f"Based on the attached data and your peers' opinions, provide your answer and a concise explanation (3-5 sentences).\n"
                        f"Format: 'Answer: [Yes/No] because [Reasoning]'\n\n"
                        f"Answer: {round_decisions[k][idx_in_round]}" # Force consistency with the probe
                    )
                    gen_prompts.append([{"role": "user", "content": [{"type": "text", "text": txt}] + content}])
                
                # Generate reasoning (continuation)
                continuations = generate_response(gen_prompts, model, processor, max_tokens)
                
                # Update reasoning state
                for i, global_idx in enumerate(new_batch_indices):
                    idx_in_round = indices_map[global_idx]
                    decision_prefix = round_decisions[k][idx_in_round]
                    current_reasoning[k][global_idx] = f"Answer: {decision_prefix} {continuations[i]}"

        # Update active list
        batch_indices = new_batch_indices

    return _format_batch_results(batch, final_texts, final_probs, modality_requests)

def _run_debate_multimodal_batch(batch, model, processor, args):
    """
    Consensus Debate (MultiModal) - 'Answer First' Variation.
    Uses 'temperature=0.7' for diversity in reasoning, but probes logits for consensus.
    """
    allowed_mods = _parse_allowed_modalities(args)
    max_tokens = 100
    n_rounds = getattr(args, 'debate_rounds', 3)
    n_agents = 4
    
    # State
    current_reasoning = [["No prior reasoning."] * len(batch) for _ in range(n_agents)]
    batch_indices = list(range(len(batch)))
    final_probs = np.zeros(len(batch))
    final_texts = [""] * len(batch)
    modality_requests = [{'patient_summary': 1, 'ehr_timeseries': 1, 'radiology_report': 1, 'cxr': 1}] * len(batch)

    print(f"   [Debate MultiModal] Running {n_rounds} rounds with {n_agents} agents...")

    for r in range(n_rounds):
        if not batch_indices: break
        
        round_probs = []
        round_decisions = []

        # A. Collect Votes (Probe)
        for k in range(n_agents):
            prompts = []
            for i in batch_indices:
                # Build Peer Context
                peer_txt = ""
                if r > 0:
                    peer_txt = "\n--- PEER ANALYSES (Previous Round) ---\n"
                    for j in range(n_agents):
                        if k != j: peer_txt += f"Agent {j+1}: {current_reasoning[j][i]}\n"
                
                content, _ = _build_prompt_content(batch[i], batch[i].get('ehr_text', ""), allowed_mods, prompt_type="data_only")
                
                txt = (
                    f"You are expert ICU risk prediction model {k+1}. This patient was just discharged.\n"
                    f"--- TASK ---\n"
                    f"Does this patient die within one year post discharge from the ICU?\n"
                    f"Below are your peers' opinions.\n"
                    f"{peer_txt}\n"
                    f"Based on the attached data and your peers' opinions, provide your answer and a concise explanation (3-5 sentences).\n"
                    f"Format: 'Answer: [Yes/No] because [Reasoning]'\n\n"
                    f"Answer:" # End prompt here
                )
                prompts.append([{"role": "user", "content": [{"type": "text", "text": txt}] + content}])
            
            # --- DEBUG INSERTION ---
            _print_debug_sample(args, batch, prompts, tag=f"Debate MM Round {r+1} - Agent {k+1}")

            # Use Greedy probe for the 'vote' to be stable
            decisions, probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
            round_probs.append(probs)
            round_decisions.append(decisions)

        # B. Check Consensus
        new_batch_indices = []
        round_probs_T = np.array(round_probs).T
        
        for idx_in_active, global_idx in enumerate(batch_indices):
            p_vals = round_probs_T[idx_in_active]
            
            # Strict Consensus Check
            all_yes = np.all(p_vals > 0.5)
            all_no = np.all(p_vals <= 0.5)
            
            is_consensus = all_yes or all_no
            is_last_round = (r == n_rounds - 1)

            if is_consensus or is_last_round:
                avg_prob = np.mean(p_vals)
                final_probs[global_idx] = avg_prob
                status = "CONSENSUS" if is_consensus else "MAX_ROUNDS"
                log = f"[{status} Round {r+1}]\nVotes: {p_vals}"
                final_texts[global_idx] = log
            else:
                new_batch_indices.append(global_idx)

        # C. Generate Reasoning (Sampled)
        if new_batch_indices:
            indices_map = {g_idx: i for i, g_idx in enumerate(batch_indices)}
            
            for k in range(n_agents):
                gen_prompts = []
                for global_idx in new_batch_indices:
                    idx_in_round = indices_map[global_idx]
                    
                    # Rebuild prompt
                    peer_txt = ""
                    if r > 0:
                        peer_txt = "\n--- PEER ANALYSES ---\n"
                        for j in range(n_agents):
                            if k != j: peer_txt += f"Agent {j+1}: {current_reasoning[j][global_idx]}\n"
                    
                    content, _ = _build_prompt_content(batch[global_idx], batch[global_idx].get('ehr_text', ""), allowed_mods, prompt_type="data_only")
                    
                    txt = (
                        f"You are expert ICU risk prediction model {k+1}. This patient was just discharged.\n"
                        f"--- TASK ---\n"
                        f"Does this patient die within one year post discharge from the ICU?\n"
                        f"Below are your peers' opinions.\n"
                        f"{peer_txt}\n"
                        f"Based on the attached data and your peers' opinions, provide your answer and a concise explanation (3-5 sentences).\n"
                        f"Format: 'Answer: [Yes/No] because [Reasoning]'\n\n"
                        f"Answer: {round_decisions[k][idx_in_round]}" # Pre-fill with the vote
                    )
                    gen_prompts.append([{"role": "user", "content": [{"type": "text", "text": txt}] + content}])
                
                # Sample reasoning to maintain diversity (Multimodal specific)
                # We use the previous vote as a prefix, but sample the explanation
                continuations = generate_response(gen_prompts, model, processor, max_tokens, do_sample=True, temperature=0.7)
                
                for i, global_idx in enumerate(new_batch_indices):
                    idx_in_round = indices_map[global_idx]
                    prefix = round_decisions[k][idx_in_round]
                    current_reasoning[k][global_idx] = f"Answer: {prefix} {continuations[i]}"

        batch_indices = new_batch_indices

    return _format_batch_results(batch, final_texts, final_probs, modality_requests)

def _run_meta_prompting_medical_batch(batch, model, processor, args):
    """
    Meta-Prompting Batch Runner.
    1. Runs the reasoning loop for every patient to build context.
    2. Debug prints the full history (System -> Experts -> Final State).
    3. Calculates implicit probabilities on the final token.
    """
    allowed_mods = _parse_allowed_modalities(args)
    final_prompts = []
    modality_requests_list = []

    print(f"   [Meta-Prompting] Reasoning loop for {len(batch)} patients...")

    for patient in batch:
        scaffold = MedicalMetaScaffolding(
            model=model, 
            processor=processor, 
            patient_data=patient, 
            allowed_modalities=allowed_mods,
            max_tokens=args.max_new_tokens
        )
        
        # This returns the history exactly at the point of decision (or timeout)
        # It will end with the model's own thought process, ready for the next token.
        history = scaffold.run_meta_loop()
        
        # Track usage stats
        trace_str = str(history)
        modality_requests_list.append({
            'patient_summary': 1,
            'ehr_timeseries': 1 if "Expert EHR" in trace_str else 0,
            'cxr': 1 if "Expert CXR" in trace_str else 0,
            'radiology_report': 1 if "Expert Report" in trace_str else 0
        })

        final_prompts.append(history)

    # --- DEBUGGING HOOK ---
    # This will print the full conversation log for the first few patients in the batch
    _print_debug_sample(args, batch, final_prompts, tag="Meta-Prompt Final Trace")

    # --- Phase 2: Batch Probability Calculation ---
    print(f"   [Meta-Prompting] Batch calculating implicit probabilities...")
    texts, exp_probs = generate_yes_no_probability(final_prompts, model, processor, max_tokens=1)

    return _format_batch_results(batch, texts, exp_probs, modality_requests_list)

def _run_reconcile_multi_model_batch(batch, main_model, main_processor, args):
    
    print("   [ReConcile] 🗑️  Clearing initial main_model to prevent OOM...")
    del main_model
    del main_processor
    """
    True ReConcile Implementation: 
    Automatically cycles through a defined set of diverse physical models.
    """
    # 1. Define Default Models (if none provided via CLI)
    default_models = [
        "OpenGVLab/InternVL2-8B",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "AIDC-AI/Ovis2-8B"
    ]

    # 2. Determine Model List
    if hasattr(args, 'reconcile_models') and args.reconcile_models:
        model_ids = [m.strip() for m in args.reconcile_models.split(',')]
    else:
        print(f"   [ReConcile] No model list provided. Running ALL default models: {default_models}")
        model_ids = default_models
    
    n_rounds = getattr(args, 'debate_rounds', 2)
    max_tokens = getattr(args, 'max_new_tokens', 256)
    allowed_mods = _parse_allowed_modalities(args)
    
    print(f"   [ReConcile] Starting Round-Table with: {model_ids}")

    # State Storage: [Model_Index][Batch_Index] -> { 'text': str, 'conf': float, 'vote': int }
    agent_states = [[{} for _ in range(len(batch))] for _ in range(len(model_ids))]

    # --- Helper: Generate for specific agent ---
    # --- Helper: Generate for specific agent ---
    def _generate_for_agent(agent_idx, current_prompts):
        m_id = model_ids[agent_idx]
        print(f"   [ReConcile] Activating Agent: {m_id}")
        
        # Create temp args to bypass main_model argument
        temp_args = type('Args', (object,), {'model_id': m_id})()
        
        # --- FIX: DELETE OLD MODELS (DO NOT MOVE TO CPU) ---
        global MODEL_CACHE
        
        # Identify keys to remove (everything that isn't the model we need right now)
        keys_to_remove = [k for k in MODEL_CACHE.keys() if k != m_id]
        
        if keys_to_remove:
            # print(f"   [Memory] Deleting {keys_to_remove} from memory...")
            for k in keys_to_remove:
                # 1. Delete the entry from the dict
                del MODEL_CACHE[k]
            
            # 2. Force Python to release System RAM immediately
            gc.collect()
            
            # 3. Force PyTorch to release VRAM immediately
            torch.cuda.empty_cache()
        # ---------------------------------------------------

        curr_model, curr_proc = get_model_and_processor(temp_args)
        
        return generate_response(
            current_prompts, 
            curr_model, 
            curr_proc, 
            max_tokens, 
            do_sample=True,
            temperature=0.7 
        )

    # --- Round 0: Initial Independent Analysis ---
    print(f"   [ReConcile] Round 0: Gathering Initial Opinions...")
    for k, m_id in enumerate(model_ids):
        prompts = []
        for p in batch:
            # We use 'cot_reasoning' so the prompt ends with "Reasoning:" 
            # This prevents the model from answering too early.
            content, _ = _build_prompt_content(p, p.get('ehr_text', ""), allowed_mods, prompt_type="cot_reasoning")
            
            sys_prompt = (
                f"{STANDARD_SYS_TEXT} Model ID: {m_id}.\n"                "Your Goal: Answer the specific medical question below based on the patient data.\n\n"
                "--- QUESTION ---\n"
                "Does this patient die in the ICU?\n\n"
                "--- INSTRUCTIONS ---\n"
                "1. Analyze the clinical data step-by-step.\n"
                "2. Estimate your confidence (0-100%) in your prediction.\n"
                "3. Provide your final One-Word Answer (Yes or No).\n\n"
                "--- REQUIRED OUTPUT FORMAT ---\n"
                "Reasoning: <Your analysis>\n"
                "Confidence: <0-100%>\n"
                "Answer: <Yes or No>"
            )
            prompts.append([{"role": "user", "content": [{"type": "text", "text": sys_prompt}] + content}])
        
        responses = _generate_for_agent(k, prompts)
        
        for i, text in enumerate(responses):
            # --- DEBUGGING PRINT ---
            print(f"\n>>> [DEBUG RAW {m_id} Patient {i}]:\n{text}\n{'-'*40}")
            # -----------------------

            conf = _parse_generated_confidence(text)
            
            # Robust Vote Extraction (With markdown cleanup)
            clean_text = text.replace('*', '')
            if re.search(r"Answer\s*[:=]\s*Yes", clean_text, re.IGNORECASE):
                vote = 1
            elif re.search(r"\bYes\W*$", clean_text, re.IGNORECASE):
                vote = 1
            else:
                vote = 0
                
            agent_states[k][i] = {'text': text, 'conf': conf, 'vote': vote}
    # --- Rounds 1 to N: Group Discussion Loop ---
    for r in range(n_rounds):
        print(f"   [ReConcile] Round {r+1} Discussion...")
        
        for k, m_id in enumerate(model_ids):
            prompts = []
            for i in range(len(batch)):
                # 1. Compile Peer Context
                peer_context = ""
                for j, peer_id in enumerate(model_ids):
                    if k == j: continue
                    peer_state = agent_states[j][i]
                    peer_context += (
                        f"--- Agent {peer_id} (Confidence: {int(peer_state['conf']*100)}%) ---\n"
                        f"{peer_state['text']}\n\n"
                    )

                # 2. ReConcile Prompt
                reconcile_txt = (
                    f"You are {m_id}. You are in a round-table conference.\n"
                    f"--- PEER OPINIONS ---\n"
                    f"{peer_context}\n"
                    f"--- YOUR PREVIOUS STANCE ---\n"
                    f"{agent_states[k][i]['text']}\n\n"
                    f"--- TASK ---\n"
                    "Re-evaluate the question: 'Does this patient die in the ICU?'\n"
                    "1. Critically review your peers. If they have better evidence, concede.\n"
                    "2. If they are wrong, provide specific evidence to CONVINCE them.\n\n"
                    "--- REQUIRED OUTPUT FORMAT ---\n"
                    "Reasoning: <Updated analysis>\n"
                    "Confidence: <0-100%>\n"
                    "Answer: <Yes or No>"
                )
                
                content, _ = _build_prompt_content(batch[i], batch[i].get('ehr_text', ""), allowed_mods, prompt_type="cot_reasoning")
                prompts.append([{"role": "user", "content": [{"type": "text", "text": reconcile_txt}] + content}])

            responses = _generate_for_agent(k, prompts)
            
            for i, text in enumerate(responses):
                conf = _parse_generated_confidence(text)
                vote = 1 if "Answer: Yes" in text or "Answer:Yes" in text or text.strip().endswith("Yes") else 0
                agent_states[k][i] = {'text': text, 'conf': conf, 'vote': vote}
            
            _print_debug_sample(args, batch, prompts, tag=f"ReConcile R{r+1} ({m_id})")

    # --- Final Aggregation: Confidence-Weighted Voting ---
    final_probs = []
    final_texts = []
    
    for i in range(len(batch)):
        weighted_sum = 0.0
        total_confidence = 0.0
        debug_log = f"--- ReConcile Final Vote (Patient {batch[i]['stay_id']}) ---\n"
        
        for k, m_id in enumerate(model_ids):
            state = agent_states[k][i]
            # Accumulate Weighted Vote
            weighted_sum += state['vote'] * state['conf']
            total_confidence += state['conf']
            debug_log += f"[{m_id}]: Vote={state['vote']}, Conf={state['conf']:.2f}\n"

        # Calculate Final Weighted Probability
        if total_confidence == 0:
            final_prob = 0.5
        else:
            final_prob = weighted_sum / total_confidence
            
        final_probs.append(final_prob)
        final_texts.append(debug_log)

    return _format_batch_results(
        batch, 
        final_texts, 
        np.array(final_probs), 
        [{'patient_summary': 1, 'ehr_timeseries': 1, 'radiology_report': 1, 'cxr': 1}] * len(batch)
    )
    
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
    
def _run_mad_agent_batch(batch, model, processor, args):
    """
    MAD: Multi-Agent Debate - Medical Adaptation.
    Final Fix: 
    1. Feeds multimodal content (Images/EHR) to ALL agents (Debaters, Moderator, Judge).
    2. Correctly tracks and reports 'modality_requests' based on 'allowed_mods'.
    """
    allowed_mods = _parse_allowed_modalities(args)
    max_tokens = getattr(args, 'max_new_tokens', 256)
    max_rounds = 3
    
    batch_size = len(batch)
    active_mask = [True] * batch_size
    debate_histories = ["" for _ in range(batch_size)]
    
    aff_latest = [""] * batch_size
    neg_latest = [""] * batch_size
    final_transcripts = [""] * batch_size
    
    # Track per-patient modality usage for final reporting
    modality_requests_list = []
    
    # Pre-calculate modality usage for the batch
    # We do this once to check which patients actually had images available
    for p in batch:
        # We call this just to check availability logic (has_image)
        _, has_image = _build_prompt_content(p, p.get('ehr_text', ""), allowed_mods, prompt_type="standard")
        
        modality_requests_list.append({
            'patient_summary': 1 if 'ps' in allowed_mods else 0,
            'ehr_timeseries': 1 if 'ehr' in allowed_mods else 0,
            'radiology_report': 1 if 'rr' in allowed_mods else 0,
            'cxr': 1 if 'cxr' in allowed_mods and has_image else 0
        })

    print(f"   [MAD] Starting Debate (Max {max_rounds} rounds)...")

    meta_prompt = (
        "You are a medical debater. We are conducting a clinical case review in a debate format. "
        "It is not necessary to agree with the other side; our objective is to find the correct prognosis.\n"
        "Debate Topic: Does this patient die within one year post discharge from the ICU?"
    )

    for round_idx in range(max_rounds):
        active_indices = [i for i, x in enumerate(active_mask) if x]
        if not active_indices:
            break
        
        print(f"   [MAD] Round {round_idx + 1}: {len(active_indices)} active debates.")

        # --- 1. Affirmative Side (Pessimist) ---
        aff_prompts = []
        for i in active_indices:
            p = batch[i]
            # Content is filtered by allowed_mods inside this function
            content, _ = _build_prompt_content(p, p.get('ehr_text', ""), allowed_mods, prompt_type="data_only")
            
            hist = f"History of previous arguments:\n{debate_histories[i]}\n" if round_idx > 0 else ""
            sys_msg = (
                f"{meta_prompt}\n\n"
                f"{hist}"
                "You are the Affirmative Side (Pessimist).\n"
                "You argue the correct outcome is: DEATH (Yes).\n"
                "Restate this stance and provide your clinical reasons. Keep the reasoning within 2-4 sentences."
            )
            aff_prompts.append([{"role": "user", "content": [{"type": "text", "text": sys_msg}] + content}])
        
        curr_aff_outs = generate_response(aff_prompts, model, processor, max_tokens)
        for idx, i in enumerate(active_indices):
            aff_latest[i] = curr_aff_outs[idx]

        # --- 2. Negative Side (Optimist) ---
        neg_prompts = []
        for idx, i in enumerate(active_indices):
            p = batch[i]
            content, _ = _build_prompt_content(p, p.get('ehr_text', ""), allowed_mods, prompt_type="data_only")
            
            sys_msg = (
                f"{meta_prompt}\n\n"
                f"Affirmative Argument: \"{curr_aff_outs[idx]}\"\n\n"
                "You are the Negative Side (Optimist).\n"
                "You disagree with the Affirmative. You argue the correct outcome is: SURVIVAL (No).\n"
                "Provide your reasons. Keep the reasoning within 2-4 sentences."
            )
            neg_prompts.append([{"role": "user", "content": [{"type": "text", "text": sys_msg}] + content}])
        
        curr_neg_outs = generate_response(neg_prompts, model, processor, max_tokens)
        for idx, i in enumerate(active_indices):
            neg_latest[i] = curr_neg_outs[idx]

        # --- 3. Moderator (Decision) ---
        mod_prompts = []
        for idx, i in enumerate(active_indices):
            p = batch[i]
            content, _ = _build_prompt_content(p, p.get('ehr_text', ""), allowed_mods, prompt_type="data_only")
            
            transcript = (
                f"Round {round_idx+1}:\n"
                f"Affirmative (Death): {aff_latest[i]}\n"
                f"Negative (Survival): {neg_latest[i]}\n"
            )
            debate_histories[i] += transcript + "\n"
            
            txt = (
                f"You are the Moderator. Evaluate the debate against the patient data provided below.\n"
                f"Debate Transcript:\n{transcript}\n"
                "Determine if there is a clear preference for one outcome based on clinical evidence.\n"
                "If yes, output 'Answer: Yes' (Death) or 'Answer: No' (Survival).\n"
                "If the case is still ambiguous and needs more debate, output 'Continue'.\n"
                "Output:"
            )
            mod_prompts.append([{"role": "user", "content": [{"type": "text", "text": txt}] + content}])
        
        mod_outs = generate_response(mod_prompts, model, processor, max_tokens=20)
        
        # --- 4. Process Decisions ---
        for idx, i in enumerate(active_indices):
            decision = mod_outs[idx].strip()
            
            if "Continue" in decision and round_idx < max_rounds - 1:
                pass 
            else:
                active_mask[i] = False
                final_transcripts[i] = debate_histories[i] + f"\nModerator Conclusion: {decision}"

    # --- 5. Final Judge ---
    print(f"   [MAD] Generating Final Verdicts...")
    judge_prompts = []
    
    for i in range(batch_size):
        p = batch[i]
        content, _ = _build_prompt_content(p, p.get('ehr_text', ""), allowed_mods, prompt_type="data_only")
        
        if final_transcripts[i] == "":
            final_transcripts[i] = debate_histories[i]
            
        txt = (
            f"Review the clinical debate history:\n{final_transcripts[i]}\n"
            "--- FINAL VERDICT ---\n"
            "Based on the debate and the patient data below, what is the correct prognosis?\n"
            "Does this patient die within one year post discharge from the ICU? Answer only Yes or No.\n"
            "Answer:"
        )
        judge_prompts.append([{"role": "user", "content": [{"type": "text", "text": txt}] + content}])

    texts, exp_probs = generate_yes_no_probability(judge_prompts, model, processor, max_tokens=1)
    
    return _format_batch_results(batch, final_transcripts, exp_probs, modality_requests_list)
    
def _run_traj_coa_agent_batch(batch, model, processor, args):
    """
    Traj-CoA: Sequence-based EHR Memory.
    - Divides EHR stream into sequential steps.
    - Maintains a 'Running Clinical Impression' (EHRMem).
    - Uses standard prompt builder for final decision.
    """
    allowed_mods = _parse_allowed_modalities(args)
    max_tokens = getattr(args, 'max_new_tokens', 256)
    num_steps = 4 
    
    batch_size = len(batch)
    final_memories = ["No significant events."] * batch_size
    modality_requests_list = []

    # --- PHASE 1: Sequential Workers (Update EHRMem) ---
    if 'ehr' in allowed_mods:
        print(f"   [Traj-CoA] Processing EHR timeline in {num_steps} sequential steps...")
        
        patient_chunks_map = [] 
        for p in batch:
            c = _chunk_ehr_sequence(p.get('ehr_text', ""), num_chunks=num_steps)
            patient_chunks_map.append(c)
            
        current_memories = ["Patient discharged from ICU. Initial observation."] * batch_size
        
        for step in range(num_steps):
            print(f"   [Traj-CoA] Step {step + 1}/{num_steps}...")
            
            prompts = []
            for i in range(batch_size):
                chunk_text = patient_chunks_map[i][step]
                prev_mem = current_memories[i]
                
                # Worker Prompt: Focused on updating memory
                # Removed specific time duration mentions
                txt = (
                    f"You are an ICU Resident tracking a patient's admission.\n\n"
                    f"--- RUNNING CLINICAL IMPRESSION (Previous) ---\n"
                    f"{prev_mem}\n\n"
                    f"--- NEW VITALS/LABS (Chronological Block {step+1}/{num_steps}) ---\n"
                    f"{chunk_text}\n\n"
                    f"--- TASK ---\n"
                    f"Update the Clinical Impression based on this new block of data. Keep it within 2-4 sentences.\n"
                    f"Note any deterioration, stability, or response to treatment.\n"
                    f"Updated Impression:"
                )
                prompts.append([{"role": "user", "content": [{"type": "text", "text": txt}]}])
            
            updates = generate_response(prompts, model, processor, max_tokens=150)
            for i in range(batch_size):
                current_memories[i] = updates[i]
        
        final_memories = current_memories

    # --- PHASE 2: Lead Agent (Synthesis) ---
    print(f"   [Traj-CoA] Lead Agent synthesizing Prognosis...")
    final_prompts = []

    for i, p in enumerate(batch):
        # 1. Replace the raw 'ehr_text' with our distilled 'EHRMem'
        original_ehr = p.get('ehr_text', "")
        p['ehr_text'] = f"--- CLINICAL COURSE SUMMARY (EHRMem) ---\n{final_memories[i]}"
        
        # 2. Use Standard Helper
        # We use prompt_type="standard" (default) which AUTOMATICALLY adds:
        # "Does this patient die in the ICU? Answer only using one word - Yes or No"
        content, has_image = _build_prompt_content(p, p['ehr_text'], allowed_mods, prompt_type="standard")
        
        # 3. Restore original text
        p['ehr_text'] = original_ehr

        # 4. Add to batch
        # Optionally add a system instruction at the top if desired, otherwise just use content
        sys_header = [{"type": "text", "text": f"{STANDARD_SYS_TEXT} Review the summarized clinical course and admission data.\n\n"}]
        final_prompts.append([{"role": "user", "content": sys_header + content}])

        modality_requests_list.append({
            'patient_summary': 1 if 'ps' in allowed_mods else 0,
            'ehr_timeseries': 1 if 'ehr' in allowed_mods else 0,
            'radiology_report': 1 if 'rr' in allowed_mods else 0,
            'cxr': 1 if 'cxr' in allowed_mods and has_image else 0
        })

    texts, exp_probs = generate_yes_no_probability(final_prompts, model, processor, max_tokens=1)
    
    return _format_batch_results(batch, final_memories, exp_probs, modality_requests_list)
    
def _run_agenticds_batch(batch, model, processor, args):
    """
    Inference Phase: Narrative -> Experts (w/ Rules) -> Judge (w/ Rules)
    """
    if not hasattr(args, 'rulebook_dir'): args.rulebook_dir = args.output_dir
    narrative_rules, diagnostic_rules = _load_rulebooks(args.rulebook_dir)
    allowed_mods = _parse_allowed_modalities(args)
    batch_size = len(batch)
    
    # 1. Narrative Generation
    narrative_prompts = []
    nr_str = "\n".join(narrative_rules)
    for p in batch:
        sys = f"You are a Medical Scribe. Synthesize a clinical narrative following these rules for the attached data:\n{nr_str}\n\n"
        content, _ = _build_prompt_content(p, p.get('ehr_text', ''), allowed_mods, prompt_type="data_only")
        narrative_prompts.append([{"role": "user", "content": [{"type": "text", "text": sys}] + content}])
    
    generated_narratives = generate_response(narrative_prompts, model, processor, max_tokens=256)
    
    # 2. Expert Analysis
    expert_analyses = ["" for _ in range(batch_size)]
    for mod in ['ehr', 'cxr', 'rr', 'ps']:
        if mod not in allowed_mods: continue
        mod_rules = diagnostic_rules.get(mod, ["Analyze the data."])
        mr_str = "\n".join(mod_rules)
        
        prompts = []
        for i, p in enumerate(batch):
            sys = (
                f"You are a {mod.upper()} Expert.\n"
                f"Rules:\n{mr_str}\n\n"
                f"Context (Patient Narrative):\n{generated_narratives[i]}\n\n"
                f"Task: Analyze the attached modality data for mortality within one year post discharge (3-5 sentences).\n"
            )
            content, _ = _build_prompt_content(p, p.get('ehr_text', ''), [mod], prompt_type="data_only")
            prompts.append([{"role": "user", "content": [{"type": "text", "text": sys}] + content}])
            
        outputs = generate_response(prompts, model, processor, max_tokens=1000)
        for i in range(batch_size): expert_analyses[i] += f"[{mod.upper()} Expert]: {outputs[i]}\n"

    # 3. Multimodal Judge
    final_prompts = []
    mm_rules = diagnostic_rules.get('multimodal', ["Synthesize all data."])
    mmr_str = "\n".join(mm_rules)
    
    for i, p in enumerate(batch):
        sys = (
            f"{STANDARD_SYS_TEXT} You are acting as the Lead Diagnostician.\n"
            f"Rules:\n{mmr_str}\n\n"
            f"--- SYNTHESIZED NARRATIVE ---\n{generated_narratives[i]}\n\n"
            f"--- EXPERT ANALYSES ---\n{expert_analyses[i]}\n\n"
            f"--- TASK ---\n"
            f"Review the full patient data below and the expert opinions above. Does this patient die within one year post discharge from the ICU?\n"
            f"Answer:"
        )
        content, _ = _build_prompt_content(p, p.get('ehr_text', ''), allowed_mods, prompt_type="data_only")
        final_prompts.append([{"role": "user", "content": [{"type": "text", "text": sys}] + content}])

    texts, probs = generate_yes_no_probability(final_prompts, model, processor, max_tokens=1)
    return _format_batch_results(batch, texts, probs, [{'patient_summary': 1}] * batch_size)


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
    elif agent_setup_name == 'MAD':
        return _run_mad_agent_batch(batch, model, processor, args)
    elif agent_setup_name == 'Debate_Unimodal':
        return _run_debate_unimodal_batch(batch, model, processor, args)
    elif agent_setup_name == 'Debate_Multimodal':
        return _run_debate_multimodal_batch(batch, model, processor, args)
    elif agent_setup_name == 'MetaPrompting':
        return _run_meta_prompting_medical_batch(batch, model, processor, args)
    elif agent_setup_name == 'ReConcile':
        return _run_reconcile_multi_model_batch(batch, model, processor, args)
    elif agent_setup_name == 'DualAgent':
        return _run_dual_agent_batch(batch, model, processor, args)
    elif agent_setup_name == 'Traj_COA_multimodal':
        return _run_traj_coa_agent_batch(batch, model, processor, args)
    elif agent_setup_name == 'AgentiCDS':
        return _run_agenticds_batch(batch, model, processor, args)
    else:
        raise ValueError(f"Unknown agent_setup: '{agent_setup_name}'.")