import json
import os
import glob
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BASE_DIR = "/scratch/baj321/MedAgent/results/ablations/Llava"

# The specific folders you listed
FOLDERS = [
    "mv-ps-cxr-rr",
    "mv-ps-cxr",
    "zeroshot-ps-cxr-rr",
    "zeroshot-ps-rr",
    "zeroshot-ps-cxr",
    "zeroshot-ps-ehr"
]

# ---------------------------------------------------------
# ECE Function
# ---------------------------------------------------------
def expected_calibration_error(samples, true_labels, M=5):
    # Uniform binning approach
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
# Helper: Find the results file
# ---------------------------------------------------------
def find_results_file(directory):
    if not os.path.exists(directory):
        return None
    
    # Case-insensitive search for *results*.json
    # This covers None_Results.json, None_results.json, results.json, etc.
    files = os.listdir(directory)
    candidates = [f for f in files if "results" in f.lower() and f.endswith(".json")]
    
    if not candidates:
        return None
    
    # Return the full path of the first match
    return os.path.join(directory, candidates[0])

# ---------------------------------------------------------
# Processing Logic
# ---------------------------------------------------------
def get_ece_for_folder(folder_name):
    full_path = os.path.join(BASE_DIR, folder_name)
    filepath = find_results_file(full_path)
    
    if not filepath:
        return "File Not Found"

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return f"Error: {e}"

    if not data or not isinstance(data, list):
        return "Error: Not a list"

    true_labels = []
    sample_probs = []

    # ASSUMPTION: 'None' in filename implies Mortality task
    gt_key = "in_hospital_mortality_48hr"
    prob_key = "mortality_probability"

    for entry in data:
        if not isinstance(entry, dict): continue
        
        # Check keys
        if "ground_truth" not in entry or gt_key not in entry["ground_truth"]:
            continue
        if "predictions" not in entry or prob_key not in entry["predictions"]:
            continue

        label = entry["ground_truth"][gt_key]
        p_class_1 = float(entry["predictions"][prob_key])

        # Construct Sample [Prob_Class_0, Prob_Class_1]
        p_class_0 = 1.0 - p_class_1
        sample_probs.append([p_class_0, p_class_1])
        true_labels.append(label)

    if not true_labels:
        return "No valid entries"

    samples_np = np.array(sample_probs)
    true_labels_np = np.array(true_labels)

    return expected_calibration_error(samples_np, true_labels_np)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    print(f"{'Ablation Folder':<30} | {'ECE Score':<15}")
    print("-" * 50)

    for folder in FOLDERS:
        score = get_ece_for_folder(folder)
        
        if isinstance(score, float):
            print(f"{folder:<30} | {score:.4f}")
        else:
            print(f"{folder:<30} | {score}")

if __name__ == "__main__":
    main()