import os
import torch
import json
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm

# --- LOCAL IMPORTS ---
from agent_architectures import (
    get_model_and_processor, 
    _prepare_inputs_for_vlm, 
    _build_prompt_content, 
    _parse_allowed_modalities,
    generate_yes_no_probability 
)
from datasets.data_utils import get_data_loader 
from evaluation import evaluate_predictions

# --- UNIFIED PROMPT CONSTANTS ---
STANDARD_SYS_TEXT = "You are an expert ICU risk prediction model. This patient was just admitted."
STANDARD_SYS_MSG = {"type": "text", "text": STANDARD_SYS_TEXT + "\n\n"}

class TrainingDataset(Dataset):
    """
    Wraps the loaded patient data (dicts) into the format expected by the model for training.
    """
    def __init__(self, data_list, allowed_modalities):
        self.data = data_list
        self.allowed_modalities = allowed_modalities

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data[idx]
        
        # 1. Build User Prompt (Standard Zero-Shot format)
        user_content_list, has_image = _build_prompt_content(
            patient, 
            patient.get('ehr_text', "EHR Data Not Available"), 
            self.allowed_modalities, 
            is_training_example=False, 
            prompt_type="standard" 
        )
        
        full_user_content = [STANDARD_SYS_MSG] + user_content_list

        # 2. Get Ground Truth
        label_int = patient['labels']['in_hospital_mortality_48hr'] 
        answer_text = "Yes" if label_int == 1 else "No"

        # 3. Construct Training Conversation (User -> Assistant)
        conversation = [
            {"role": "user", "content": full_user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
        ]
        return conversation

def collate_fn_vlm(batch, processor, device="cpu"):
    inputs = _prepare_inputs_for_vlm(batch, processor, device)
    return inputs

def run_evaluation_loop(model, processor, test_loader, args):
    """
    Runs inference and transforms data to use the shared evaluate_predictions function.
    """
    print(f"\n--- Starting Evaluation on Test Set ---")
    model.eval()
    
    allowed_mods = _parse_allowed_modalities(args)
    formatted_results = [] 
    
    # Iterate through the pre-loaded MemoryDataset using the DataLoader
    for batch in tqdm(test_loader, desc="Evaluating"):
        prompts = []
        batch_patients = [] 
        
        # 1. Prepare Prompts
        for p in batch:
            user_content, _ = _build_prompt_content(
                p, p.get('ehr_text', ""), allowed_mods, is_training_example=False, prompt_type="standard"
            )
            prompts.append([{"role": "user", "content": [STANDARD_SYS_MSG] + user_content}])
            batch_patients.append(p)
            
        # 2. Run Inference
        try:
            _, probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
            
            # 3. Format results for the shared evaluation function
            for i, p in enumerate(batch_patients):
                prob = probs[i]
                pred_label = 1 if prob > 0.5 else 0
                
                res_dict = {
                    'subject_id': p.get('subject_id', 'unknown'),
                    'stay_id': p.get('stay_id', 'unknown'),
                    'ground_truth': {
                        'in_hospital_mortality_48hr': p['labels']['in_hospital_mortality_48hr']
                    },
                    'predictions': {
                        'in_hospital_mortality_48hr': pred_label,
                        'mortality_probability': prob,
                        'mortality_probability_text': f"{prob:.4f}"
                    },
                    'modality_requests': {
                        m: True for m in allowed_mods if m in p.get('available_modalities', [])
                    },
                    'modality_availability': {
                        'cxr': 'cxr' in p.get('available_modalities', []),
                        'radiology_report': 'radiology_report' in p.get('available_modalities', [])
                    }
                }
                formatted_results.append(res_dict)
            
        except Exception as e:
            print(f"Error in evaluation batch: {e}")

    # 4. Call the shared evaluation function
    print("\n--- Calculating Metrics ---")
    metrics = evaluate_predictions(formatted_results, output_csv_path=os.path.join(args.output_dir, "eval_predictions.csv"))
    
    # 5. Print Summary
    m_metrics = metrics['in_hospital_mortality_metrics']
    print(f"\nAccuracy: {m_metrics['accuracy']:.4f} ({m_metrics['accuracy_ci_low']:.4f} - {m_metrics['accuracy_ci_high']:.4f})")
    print(f"AUC:      {m_metrics['auroc']:.4f} ({m_metrics['auroc_ci_low']:.4f} - {m_metrics['auroc_ci_high']:.4f})")
    print(f"AUPRC:    {m_metrics['auprc']:.4f} ({m_metrics['auprc_ci_low']:.4f} - {m_metrics['auprc_ci_high']:.4f})")
    
    # Save full metrics to JSON
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Only switch back to train mode if we are actually training
    if args.mode == "train":
        model.train()

def main():
    parser = argparse.ArgumentParser()
    
    # --- New Arguments ---
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], 
                        help="Mode: 'train' to finetune, 'eval' to load and test.")
    parser.add_argument("--lora_adapter_path", type=str, default=None, 
                        help="Path to trained LoRA adapter (e.g. output/checkpoint-epoch-3). Required if mode=eval and you want to test a finetuned model.")

    # --- Existing Arguments ---
    parser.add_argument("--model_id", type=str, required=True, help="Base Model ID (e.g. Qwen/Qwen2-VL...)")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--modalities", type=str, default="ehr-cxr-rr-ps")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Base Model & Processor
    print(f"Loading base model: {args.model_id}")
    model, processor = get_model_and_processor(args)
    
    if "Ovis" in args.model_id:
        import types
        def get_input_embeddings_patch(self):
            return self.llm.get_input_embeddings()
        model.get_input_embeddings = types.MethodType(get_input_embeddings_patch, model)
    
    # --- EVALUATION MODE ---
    if args.mode == "eval":
        print("\n--- Running in EVALUATION Mode ---")
        
        # Load LoRA Adapter if provided
        if args.lora_adapter_path:
            print(f"Loading LoRA adapter from: {args.lora_adapter_path}")
            model = PeftModel.from_pretrained(model, args.lora_adapter_path)
            model.merge_and_unload() # Optional: Merge for potentially faster inference
        else:
            print("[Info] No LoRA adapter path provided. Running Zero-Shot / Base Model evaluation.")

        # Load Test Data
        print(f"\n[Data] Loading Test Data from: {args.test_data_path}")
        test_loader = get_data_loader(args.test_data_path, args.batch_size, args.num_workers)
        
        # Run Evaluation
        run_evaluation_loop(model, processor, test_loader, args)
        return # Exit after eval

    # --- TRAINING MODE ---
    print("\n--- Running in TRAINING Mode ---")
    model.enable_input_require_grads()
    
    if "InternVL" in args.model_id:
        # InternLM2 architecture naming convention
        target_modules = ["wqkv", "wo", "w1", "w2", "w3"]
    else:
        # Standard Llama/Qwen/Ovis naming convention
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    print(f"Setting up LoRA configuration for {args.model_id}...")
    print(f"Targeting modules: {target_modules}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=target_modules  # <--- PASS THE DYNAMIC LIST HERE
    )
    
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Data
    print(f"\n[Data] Loading Training Data from: {args.train_data_path}")
    raw_train_loader = get_data_loader(args.train_data_path, args.batch_size, args.num_workers)
    train_data_list = raw_train_loader.dataset.data 
    
    print(f"\n[Data] Loading Test Data from: {args.test_data_path}")
    test_loader = get_data_loader(args.test_data_path, args.batch_size, args.num_workers)

    allowed_mods = _parse_allowed_modalities(args)
    train_dataset = TrainingDataset(train_data_list, allowed_mods)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn_vlm(b, processor, device="cpu") 
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if getattr(model, "device", None) and str(model.device) != "meta":
        device = model.device

    model.train()
    
    # Identify Model Type for Training Logic
    is_internvl = "Intern" in args.model_id
    is_ovis = "Ovis" in args.model_id
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for step, batch_inputs in enumerate(progress_bar):
            batch_inputs = batch_inputs.to(device)
            input_ids = batch_inputs["input_ids"]
            
            # --- FIX 1: Handle Missing Pixel Values for Ovis/InternVL ---
            # If batch is text-only, these models still demand pixel_values arguments
            if "pixel_values" not in batch_inputs or batch_inputs["pixel_values"] is None:
                if is_internvl:
                    # InternVL: Needs 4D Dummy Tensor + Image Flags (0)
                    batch_inputs["pixel_values"] = torch.zeros((len(input_ids), 3, 448, 448), device=device, dtype=model.dtype)
                    batch_inputs["image_flags"] = torch.zeros((len(input_ids), 1), device=device, dtype=torch.long)
                elif is_ovis:
                    # Ovis: Needs List of Nones
                    batch_inputs["pixel_values"] = [None] * len(input_ids)

            # --- Label Masking ---
            labels = input_ids.clone()
            labels[:, :] = -100 
            labels[:, -2:] = input_ids[:, -2:]
            
            # --- FIX 2: Explicit Forward Calls ---
            if is_ovis:
                # Ovis is strict about arguments; avoid passing unexpected kwargs
                outputs = model(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    pixel_values=batch_inputs["pixel_values"],
                    labels=labels
                )
            elif is_internvl:
                # InternVL needs image_flags which might not be in **batch_inputs if we didn't add them above
                # If they were added in Fix 1, **batch_inputs covers it.
                # If real images exist, 'image_flags' should already be there from processor.
                outputs = model(**batch_inputs, labels=labels)
            else:
                # Standard (Qwen, etc.)
                outputs = model(**batch_inputs, labels=labels)
            
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            
            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Average Epoch Loss: {avg_loss:.4f}")
        
        save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)

        # Uncomment to evaluate every epoch:
        # run_evaluation_loop(model, processor, test_loader, args)

    print("\nTraining Complete. Switching to Inference Mode...")
    run_evaluation_loop(model, processor, test_loader, args)

if __name__ == "__main__":
    main()