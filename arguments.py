import argparse
import os

def args_parser():
    """
    Parses and returns command-line arguments for the multi-agent simulation.
    """
    parser = argparse.ArgumentParser(
        description='Arguments for Multi-Agent Clinical Simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core Pipeline Arguments (Required by main.py) ---
    group_core = parser.add_argument_group('Core Pipeline Settings')
    group_core.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help="Directory to save detailed results and evaluation summaries."
    )
    group_core.add_argument(
        '--agent_setup',
        type=str,
        required=True,
        help="Name of the agent setup to initialize and run."
    )
    group_core.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    # --- DataLoader Settings ---
    group_loader = parser.add_argument_group('DataLoader Settings')
    group_loader.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Number of samples to process in a single batch."
    )
    group_loader.add_argument(
        '--num_workers',
        type=int,
        default=min(os.cpu_count(), 4),
        help="Number of worker processes for data loading."
    )
    
    
    group_llm = parser.add_argument_group('LLM Configuration')
    group_llm.add_argument(
        '--model_id', 
        type=str, 
        default='HuggingFaceM4/idefics2-8b', 
        help='The Hugging Face model ID to load for the agents.'
    )
    group_llm.add_argument(
        '--max_new_tokens', 
        type=int, 
        default=128, 
        help='Maximum number of new tokens to generate for each response.'
    )
    
    # --- Legacy & Model-Specific Arguments ---
    group_legacy = parser.add_argument_group('Legacy Model & Data Paths')
    group_legacy.add_argument('--data_path', type=str, default=None, help="Path to the JSON Lines (.jsonl) file containing the test patient dataset.")
    group_legacy.add_argument('--few_shot_data_path', type=str, default=None)
    group_legacy.add_argument('--train_data_path', type=str, default=None)
    group_legacy.add_argument('--val_data_path', type=str, default=None)
    group_legacy.add_argument('--calibrate_majority_vote', type=str, default=None)
    group_legacy.add_argument('--rulebook_dir', type=str, default=None)
    group_legacy.add_argument('--task', type=str, default=None)
    group_legacy.add_argument('--train_rules', type=str, default=None)
    group_legacy.add_argument('--max_train_samples', type=int, default=None)
    group_legacy.add_argument('--mimic_notes_dir', type=str, default=None)
    group_legacy.add_argument('--consistency_samples', type=int, default=5)
    group_legacy.add_argument('--num_shots', type=int, default=None)
    group_legacy.add_argument('--refine_iterations', type=int, default=None)
    group_legacy.add_argument('--lr', type=float, default=0.01)
    group_legacy.add_argument('--debate_rounds', type=int, default=None)
    group_legacy.add_argument('--modalities', default="ehr-cxr-rr-ps", type=str, help='Specify the desired data modalities')
    group_legacy.add_argument('--load_cxr', type=str, default=None, help='Path to pretrained CXR model state')
    group_legacy.add_argument('--load_ehr', type=str, default=None, help='Path to pretrained EHR model state')
    group_legacy.add_argument('--load_rr', type=str, default=None, help='Path to pretrained RR model state')
    group_legacy.add_argument('--ablation', type=str, default=None, help='Specify an ablation study to run')
    group_legacy.add_argument('--normalizer_state', type=str, default=None, help='Path to a state file of a normalizer.')
    group_legacy.add_argument('--ehr_data_dir', type=str, default='/scratch/fs999/shamoutlab/data/mimic-iv-extracted', help='Path to supplementary EHR data directory')
    group_legacy.add_argument('--cxr_data_dir', type=str, default='/scratch/fs999/shamoutlab/data/physionet.org/files/mimic-cxr-jpg/2.0.0', help='Path to supplementary CXR data directory')
    parser.add_argument('--debug_samples', type=int, default=0, help='Number of samples to print for debugging.')
    return parser