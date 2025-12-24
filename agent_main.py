import os
import json
import torch
import numpy as np
import random
import multiprocessing

# --- Local Module Imports ---
from arguments import args_parser
from datasets.data_utils import get_data_loader
from simulation import run_simulation
from evaluation import evaluate_predictions

def set_seed(seed_value: int):
    """Set seed for reproducibility across all relevant libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

def main():
    """
    Main entry point for the multi-agent simulation.
    Orchestrates data loading, simulation execution, evaluation, and result saving.
    """
    # 1. Load arguments from the dedicated arguments file
    args = args_parser().parse_args()

    # --- OPTIMIZATION: Auto-set workers for Pre-Loading phase ---
    # This determines how many CPU cores will be used to load images/CSVs into RAM 
    # at startup. It does NOT affect the simulation loop (which is 0-latency from RAM).
    if args.num_workers == 0:
        # Heuristic: Use available cores, capped at 16 to prevent diminishing returns
        args.num_workers = min(16, os.cpu_count() or 1)
        print(f"[System] Auto-setting num_workers to {args.num_workers} for parallel data loading.")
    
    print("--- Arguments Loaded ---")
    print(json.dumps(vars(args), indent=4))
    print("-" * 26)

    # 2. Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # 3. Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")

    # 4. Get the data loader (Triggers the RAM Pre-Loading)
    print(f"\n[Status] Initializing Data Loader from: {args.data_path}")
    test_loader = get_data_loader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 5. Run the simulation
    # The run_simulation function will iterate through the data_loader (now from RAM),
    # invoke the specified agent architecture for each batch, and collect results.
    print(f"\nStarting simulation with agent setup: '{args.agent_setup}'...")
    all_results = run_simulation(test_loader, args)
    print("...Simulation complete.")

    # 6. Evaluate the collected results
    print("\nEvaluating predictions...")
    if not all_results:
        print("Warning: No results were generated from the simulation. Exiting.")
        return

    evaluation_metrics = evaluate_predictions(all_results)
    print("...Evaluation complete.")

    # 7. Save results and summary
    results_filename = f'{args.agent_setup}_detailed_results.json'
    summary_filename = f'{args.agent_setup}_evaluation_summary.json'
    
    results_path = os.path.join(args.output_dir, results_filename)
    summary_path = os.path.join(args.output_dir, summary_filename)

    # Save detailed per-patient results
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    # Save the summarized evaluation metrics
    with open(summary_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)

    print("\n" + "="*50)
    print("      Pipeline Finished Successfully")
    print("="*50 + "\n")
    print(f"Agent Setup: {args.agent_setup}")
    print("\n--- Evaluation Summary ---")
    print(json.dumps(evaluation_metrics, indent=4))
    print(f"\nDetailed results saved to: {results_path}")
    print(f"Evaluation summary saved to: {summary_path}")


if __name__ == '__main__':
    main()