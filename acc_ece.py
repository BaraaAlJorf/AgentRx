import json
import numpy as np

# ---------------------------------------------------------
# ECE Function (MATCHES YOUR ABLATION FILE)
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
        in_bin = np.logical_and(confidences > bin_lower.item(),
                                confidences <= bin_upper.item())
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return ece.item()


# ---------------------------------------------------------
# Bootstrap 95% CI for Accuracy (percentile method)
# ---------------------------------------------------------
def bootstrap_accuracy_ci(y_true, y_pred, n_bootstraps=1000, seed=42, alpha=0.05):
    """
    Bootstrap percentile CI for accuracy.
    y_true, y_pred: 1D arrays of same length
    n_bootstraps: number of bootstrap samples
    alpha: for 95% CI, alpha=0.05 -> [2.5%, 97.5%]
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    if n == 0:
        raise ValueError("No samples to compute confidence interval.")

    boot_acc = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        acc = (y_true[indices] == y_pred[indices]).mean()
        boot_acc.append(acc)

    lower = np.percentile(boot_acc, 100 * (alpha / 2.0))
    upper = np.percentile(boot_acc, 100 * (1.0 - alpha / 2.0))

    # point estimate on full data
    acc = (y_true == y_pred).mean()
    return acc, lower, upper


# ---------------------------------------------------------
# Load results.json (single file)
# ---------------------------------------------------------
with open("/scratch/baj321/MSMA/MedAgent_checkpoints/in-hospital-mortality/unimodal_dn/results.json", "r") as f:
    data = json.load(f)

# Adjust these keys if your JSON shape is different
probs = np.array(data["probabilities"], dtype=np.float32)   # p(class 1)
labels = np.array(data["ground_truth"], dtype=np.int32)     # 0/1

# Build [p0, p1] samples to match the ablation script
p_class_1 = probs
p_class_0 = 1.0 - p_class_1
samples = np.stack([p_class_0, p_class_1], axis=1)

# ---------------------------------------------------------
# Accuracy + 95% Bootstrap CI
# ---------------------------------------------------------
threshold = 0.5
preds = (p_class_1 >= threshold).astype(int)

acc, acc_low, acc_high = bootstrap_accuracy_ci(
    labels,
    preds,
    n_bootstraps=1000,
    seed=42,
    alpha=0.05,   # 95% CI
)

# ---------------------------------------------------------
# ECE (same as ablation script)
# ---------------------------------------------------------
ece = expected_calibration_error(samples, labels, M=5)

print(f"Accuracy (threshold={threshold}): {acc:.4f}")
print(f"95% bootstrap CI for accuracy: [{acc_low:.4f}, {acc_high:.4f}]")
print(f"ECE (M=5): {ece:.6f}")