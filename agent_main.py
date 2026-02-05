import os
import json
import torch
import numpy as np
import random

# --- Constant Core Imports ---
from arguments import args_parser
from datasets.data_utils import get_data_loader
from datasets.data_utils import get_train_data_loader
import simulation  # Import the module to override its internal function

def set_seed(seed_value: int):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

def main():
    args = args_parser().parse_args()
    
    # --- TASK-BASED CONDITIONAL IMPORTS ---
    if args.task == 'los':
        from agent_architectures_los import initialize_agent_setup, get_model_and_processor
        from evaluation_los import evaluate_predictions
        from rulebook_learning_los import RulebookLearner
    elif args.task == 'mortality_1yr':
        from agent_architectures_1yr import initialize_agent_setup, get_model_and_processor
        from evaluation_1yr import evaluate_predictions
        from rulebook_learning_1yr import RulebookLearner
    else:
        # Default: In-hospital mortality
        from agent_architectures import initialize_agent_setup, get_model_and_processor
        from evaluation import evaluate_predictions
        from rulebook_learning import RulebookLearner

    # Overwrite simulation's setup function so simulation.py doesn't need to change
    simulation.initialize_agent_setup = initialize_agent_setup

    # --- Standard Simulation Flow ---
    if args.num_workers == 0:
        args.num_workers = min(16, os.cpu_count() or 1)
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Found this in ARGS:",args.max_train_samples)
    
    # Phase 1: Rulebook Training
    if args.train_rules:
        model, processor = get_model_and_processor(args)
        train_loader = get_train_data_loader(args.train_data_path, args.batch_size, args.num_workers, max_train_samples= args.max_train_samples, seed=args.seed)
        learner = RulebookLearner(model, processor, args)
        learner.train_epoch(train_loader)
        learner.save_rulebooks(args.output_dir)
        args.rulebook_dir = args.output_dir
        args.agent_setup = 'AgentiCDS'
    
    if args.agent_setup == 'AgentiCDS' and not getattr(args, 'rulebook_dir', None):
        args.rulebook_dir = args.output_dir
    
    if getattr(args, 'calibrate_majority_vote', False):
        import calibrate_MV

        print(">>> Entering Calibrated Majority Vote Mode")
        
        # 1. Load Model & Processor (if not already loaded)
        if 'model' not in locals():
             model, processor = get_model_and_processor(args)

        # 2. Prepare Loaders
        # Use get_data_loader for Validation (ensures no shuffle/aug, strict evaluation mode)
        val_loader = get_data_loader(args.val_data_path, args.batch_size, args.num_workers)
        test_loader = get_data_loader(args.data_path, args.batch_size, args.num_workers)

        # 3. Handover to the calibration file
        all_results = calibrate_MV.run_calibrated_flow(model, processor, val_loader, test_loader, args)
        
        # 4. Save & Exit 
        if all_results:
            evaluation_metrics = evaluate_predictions(all_results)
            with open(os.path.join(args.output_dir, f'{args.task}_calibrated_results.json'), 'w') as f:
                json.dump(all_results, f, indent=4)
            with open(os.path.join(args.output_dir, f'{args.task}_calibrated_summary.json'), 'w') as f:
                json.dump(evaluation_metrics, f, indent=4)
        
        print("Calibration & Inference Complete.")
        return
    # --------------------------------------------------
    
    
    # Phase 2: Simulation Execution
    test_loader = get_data_loader(args.data_path, args.batch_size, args.num_workers)
    all_results = simulation.run_simulation(test_loader, args)

    # Phase 3: Evaluation
    if not all_results:
        return

    evaluation_metrics = evaluate_predictions(all_results)

    # Save outputs
    with open(os.path.join(args.output_dir, f'{args.task}_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    with open(os.path.join(args.output_dir, f'{args.task}_summary.json'), 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)

    print(f"Task {args.task} complete.")

if __name__ == '__main__':
    main()