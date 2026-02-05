import json
import os
import glob
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BASE_DIR = "/MedAgent/results"

MODEL_MAP = [
    ("final_qwen", "Qwen"),
    ("intern", "InternVL2.5"),
    ("HuaTuo", "HuaTuo"),
    ("Llava", "LlavaMed")
]

# Group 1: PS (Patient Summary)
GROUP_PS = [
    ("zeroshot-ps", "Zero-shot"),
    ("fewshot-ps", "Few-shot"),
    ("cot-ps", "CoT"),
    ("cot-sc-ps", "CoT + Self-Consistency"),
    ("sr-ps", "Self-Refinement")
]

# Group 2: Full (Single Agent)
GROUP_FULL_SINGLE = [
    ("zeroshot-full", "Zero-shot"),
    ("fewshot-full", "Few-shot"),
    ("cot-full", "CoT"),
    ("cot-sc-full", "CoT + Self-Consistency")
]

# Group 3: Full (Multi-Agent)
GROUP_FULL_MULTI = [
    ("mv-full", "Majority Vote"),
    ("debate-unimodal-full", "Debate (unimodal Agents)"),
    ("debate-full", "Debate (multimodal Agents)"),
    ("meta-full", "Meta-Prompting"),
    ("traj-full", "Traj-CoA + multimodal Agent"),
    ("MAD-full", "MAD + Judge")
]

# ---------------------------------------------------------
# ECE Function
# ---------------------------------------------------------
def expected_calibration_error(samples, true_labels, M=5):
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(samples, axis=1)
    predicted_label = np.argmax(samples, axis=1)
    accuracies = predicted_label == true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece.item()

# ---------------------------------------------------------
# Processing Logic
# ---------------------------------------------------------
def get_ece_for_path(directory, task_type):
    if not os.path.exists(directory):
        return "-"

    # Search for any json file containing "results" in the name
    search_pattern = os.path.join(directory, "*results*.json")
    files = glob.glob(search_pattern)
    
    if not files:
        # Fallback: Check if there's a file exactly named "None_results.json" or "los_results.json"
        # sometimes glob behaves strictly on some systems
        all_files = os.listdir(directory)
        candidates = [f for f in all_files if "results" in f]
        if not candidates:
            return "-"
        filepath = os.path.join(directory, candidates[0])
    else:
        filepath = files[0]

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except:
        return "ERR:JSON_LOAD"

    if not data or not isinstance(data, list):
        return "ERR:NOT_LIST"

    true_labels = []
    sample_probs = []

    # Determine keys
    gt_key = "los_7" if task_type == "los" else "in_hospital_mortality_48hr"
    prob_key = "mortality_probability"

    for entry in data:
        if not isinstance(entry, dict): continue
        
        # KEY CHECKING LOGIC
        if "ground_truth" not in entry or gt_key not in entry["ground_truth"]:
            continue
        if "predictions" not in entry or prob_key not in entry["predictions"]:
            continue

        label = entry["ground_truth"][gt_key]
        p_class_1 = float(entry["predictions"][prob_key])
        p_class_0 = 1.0 - p_class_1
        sample_probs.append([p_class_0, p_class_1])
        true_labels.append(label)

    if not true_labels:
        # --- DIAGNOSTIC PRINT ---
        # If we found the file but extracted 0 entries, print WHY.
        if len(data) > 0:
            sample = data[0]
            print(f"\n[DEBUG FAILURE] {filepath}")
            print(f"  > Looking for GT Key: '{gt_key}' inside 'ground_truth'")
            print(f"  > Looking for Prob Key: '{prob_key}' inside 'predictions'")
            print(f"  > Actual Keys in First Entry: {list(sample.keys())}")
            if "ground_truth" in sample:
                print(f"  > 'ground_truth' contents: {sample['ground_truth']}")
            else:
                print("  > 'ground_truth' key MISSING.")
        return "-"

    return expected_calibration_error(np.array(sample_probs), np.array(true_labels))

# ---------------------------------------------------------
# Main Reporting Loop
# ---------------------------------------------------------
def print_section(section_name, group_list, sub_header, task_type):
    print("=" * 60)
    print(f"SECTION: {section_name} ({task_type.upper()})")
    print("=" * 60)

    for model_dir, model_name in MODEL_MAP:
        print(f"\nModel: {model_name}")
        print(f"Type:  {sub_header}")
        print("-" * 40)
        
        for baseline_dir, baseline_name in group_list:
            if task_type == "los":
                full_path = os.path.join(BASE_DIR, "los", model_dir, baseline_dir)
            else:
                full_path = os.path.join(BASE_DIR, model_dir, baseline_dir)
            
            score = get_ece_for_path(full_path, task_type)
            
            if isinstance(score, float):
                print(f"{baseline_name:<30} : {score:.4f}")
            else:
                print(f"{baseline_name:<30} : {score}")

def main():
    tasks = ["mortality", "los"]
    
    for task in tasks:
        print("\n" + "#" * 60)
        print(f"###  TASK: {task.upper()}  ###")
        print("#" * 60 + "\n")

        # 1. PS First
        print_section("Patient Summary (PS)", GROUP_PS, "Single Agent", task)
        
        # 2. Full (Single)
        print("\n\n" + "-"*20 + " SWITCHING TO FULL MODALITY " + "-"*20 + "\n")
        print_section("Full Modality (Single)", GROUP_FULL_SINGLE, "Single Agent", task)
        
        # 3. Full (Multi)
        print_section("Full Modality (Multi)", GROUP_FULL_MULTI, "Multi-Agent", task)

if __name__ == "__main__":

    main()
