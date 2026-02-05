import numpy as np
import csv
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    average_precision_score,
    accuracy_score
)

def calculate_confidence_intervals(y_true, y_scores, y_preds, n_bootstraps=1000, seed=42):
    """
    Performs bootstrapping to calculate 95% Confidence Intervals for AUC, AUPRC, and Accuracy.
    """
    rng = np.random.RandomState(seed)
    bootstrapped_auc = []
    bootstrapped_auprc = []
    bootstrapped_acc = []

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_preds = np.array(y_preds)

    # If only one class is present, we cannot calculate ROC/PR curves
    if len(np.unique(y_true)) < 2:
        return {
            "auroc_ci": (0.0, 0.0),
            "auprc_ci": (0.0, 0.0),
            "accuracy_ci": (0.0, 0.0)
        }

    for _ in range(n_bootstraps):
        # Bootstrap sample indices
        indices = rng.randint(0, len(y_scores), len(y_scores))
        
        if len(np.unique(y_true[indices])) < 2:
            # Skip samples that don't have both positive and negative classes
            continue

        score_auc = roc_auc_score(y_true[indices], y_scores[indices])
        score_auprc = average_precision_score(y_true[indices], y_scores[indices])
        score_acc = accuracy_score(y_true[indices], y_preds[indices])

        bootstrapped_auc.append(score_auc)
        bootstrapped_auprc.append(score_auprc)
        bootstrapped_acc.append(score_acc)

    # Calculate percentiles (2.5% and 97.5%)
    results = {}
    
    if bootstrapped_auc:
        results['auroc_ci'] = (
            np.percentile(bootstrapped_auc, 2.5), 
            np.percentile(bootstrapped_auc, 97.5)
        )
    else:
        results['auroc_ci'] = (0.0, 0.0)

    if bootstrapped_auprc:
        results['auprc_ci'] = (
            np.percentile(bootstrapped_auprc, 2.5), 
            np.percentile(bootstrapped_auprc, 97.5)
        )
    else:
        results['auprc_ci'] = (0.0, 0.0)
        
    if bootstrapped_acc:
        results['accuracy_ci'] = (
            np.percentile(bootstrapped_acc, 2.5), 
            np.percentile(bootstrapped_acc, 97.5)
        )
    else:
        results['accuracy_ci'] = (0.0, 0.0)

    return results

def evaluate_predictions(all_results: list, output_csv_path: str = None) -> dict:
    """
    Calculates performance metrics for mortality ONLY.
    Now includes 95% Confidence Intervals via bootstrapping.
    """
    # --- Prepare lists for sklearn metrics ---
    y_true_mortality, y_pred_mortality = [], []
    y_scores_mortality = [] # For P(Yes) probability
    
    csv_data = []

    # --- Prepare counters for modality analysis ---
    total_patients = len(all_results)
    modality_request_counts = {'patient_summary': 0, 'ehr_timeseries': 0, 'cxr': 0, 'radiology_report': 0}
    modality_availability_counts = {'cxr': 0, 'radiology_report': 0}
    
    # --- Loop through results to aggregate data ---
    for res in all_results:
        # --- Aggregate data for metrics ---
        gt_mortality = res['ground_truth']['los_7']
        pred_mortality = res['predictions']['los_7']
        
        # Get the single P(Yes) probability and the text
        prob_mortality = res['predictions'].get('mortality_probability', 0.5)
        text_mortality = res['predictions'].get('mortality_probability_text', 'N/A')
        
        y_true_mortality.append(gt_mortality)
        y_pred_mortality.append(pred_mortality)
        y_scores_mortality.append(prob_mortality)

        # --- Aggregate data for modality analysis ---
        # Handle cases where modality_requests might not be in training data dicts
        requests = res.get('modality_requests', {})
        for modality, requested in requests.items():
            if requested and modality in modality_request_counts:
                modality_request_counts[modality] += 1
        
        availabilities = res.get('modality_availability', {})
        for modality, available in availabilities.items():
            if available and modality in modality_availability_counts:
                modality_availability_counts[modality] += 1
                
        # --- Prepare row for CSV ---
        if output_csv_path:
            row = {
                'subject_id': res.get('subject_id', ''), 
                'stay_id': res.get('stay_id', ''),
                'ground_truth_mortality': gt_mortality,
                'prediction_mortality': pred_mortality,
                'probability_mortality': prob_mortality,
                'probability_mortality_text': text_mortality
            }
            csv_data.append(row)

    # --- 1. Save detailed results to CSV (if path provided) ---
    if output_csv_path and csv_data:
        print(f"Saving detailed predictions to {output_csv_path}...")
        try:
            headers = csv_data[0].keys()
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(csv_data)
        except Exception as e:
            print(f"[{'Warning'.upper()}] Could not save CSV file: {e}")

    # --- 2. Calculate Mortality Metrics ---
    mortality_metrics = classification_report(
        y_true_mortality, y_pred_mortality, 
        target_names=['<7', '>7'], 
        output_dict=True,
        zero_division=0
    )

    # Calculate Point Estimates
    try:
        mortality_auroc = roc_auc_score(y_true_mortality, y_scores_mortality)
    except ValueError as e:
        print(f"[{'Warning'.upper()}] Could not calculate AUROC: {e}")
        mortality_auroc = 0.0
    mortality_metrics['auroc'] = mortality_auroc

    try:
        mortality_auprc = average_precision_score(y_true_mortality, y_scores_mortality)
    except ValueError as e:
        print(f"[{'Warning'.upper()}] Could not calculate AUPRC: {e}")
        mortality_auprc = 0.0
    mortality_metrics['auprc'] = mortality_auprc
    
    # --- 3. Calculate Confidence Intervals (Bootstrapping) ---
    print("Calculating Confidence Intervals (95%) via bootstrapping...")
    ci_metrics = calculate_confidence_intervals(
        y_true_mortality, 
        y_scores_mortality, 
        y_pred_mortality
    )
    
    # Add CI data to the metrics dictionary
    mortality_metrics['auroc_ci_low'] = ci_metrics['auroc_ci'][0]
    mortality_metrics['auroc_ci_high'] = ci_metrics['auroc_ci'][1]
    mortality_metrics['auprc_ci_low'] = ci_metrics['auprc_ci'][0]
    mortality_metrics['auprc_ci_high'] = ci_metrics['auprc_ci'][1]
    mortality_metrics['accuracy_ci_low'] = ci_metrics['accuracy_ci'][0]
    mortality_metrics['accuracy_ci_high'] = ci_metrics['accuracy_ci'][1]

    # --- 4. Modality Stats ---
    total_modality_requests = sum(modality_request_counts.values())
    modality_usage = {
        'total_modality_requests': total_modality_requests,
        'average_modalities_per_patient': total_modality_requests / total_patients if total_patients > 0 else 0,
        'request_counts': modality_request_counts,
    }

    # --- 5. Final Dictionary ---
    final_metrics = {
        "length_of_stay_metrics": mortality_metrics,
        "modality_usage_analysis": modality_usage,
        "total_patients_evaluated": total_patients
    }

    return final_metrics