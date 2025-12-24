import numpy as np
import csv
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    average_precision_score
)

def evaluate_predictions(all_results: list, output_csv_path: str = None) -> dict:
    """
    Calculates performance metrics for mortality ONLY.
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
        gt_mortality = res['ground_truth']['in_hospital_mortality_48hr']
        pred_mortality = res['predictions']['in_hospital_mortality_48hr']
        
        # Get the single P(Yes) probability and the text
        prob_mortality = res['predictions'].get('mortality_probability', 0.5)
        text_mortality = res['predictions'].get('mortality_probability_text', 'N/A')
        
        y_true_mortality.append(gt_mortality)
        y_pred_mortality.append(pred_mortality)
        y_scores_mortality.append(prob_mortality)

        # --- Aggregate data for modality analysis ---
        for modality, requested in res['modality_requests'].items():
            if requested:
                modality_request_counts[modality] += 1
        
        for modality, available in res['modality_availability'].items():
            if available:
                modality_availability_counts[modality] += 1
                
        # --- Prepare row for CSV ---
        if output_csv_path:
            row = {
                'subject_id': res.get('subject_id', ''), # Include Subject ID
                'stay_id': res['stay_id'],
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
    elif output_csv_path:
        print("[Warning] output_csv_path was provided, but no data was aggregated to save.")


    # --- 2. Calculate Mortality Metrics ---
    mortality_metrics = classification_report(
        y_true_mortality, y_pred_mortality, 
        target_names=['Survived', 'Mortality'], 
        output_dict=True,
        zero_division=0
    )

    # Calculate and add AUROC/AUPRC
    try:
        mortality_auroc = roc_auc_score(y_true_mortality, y_scores_mortality)
    except ValueError as e:
        print(f"[{'Warning'.upper()}] Could not calculate AUROC: {e}")
        mortality_auroc = None
    mortality_metrics['auroc'] = mortality_auroc

    try:
        mortality_auprc = average_precision_score(y_true_mortality, y_scores_mortality)
    except ValueError as e:
        print(f"[{'Warning'.upper()}] Could not calculate AUPRC: {e}")
        mortality_auprc = None
    mortality_metrics['auprc'] = mortality_auprc
    
    # --- 3. Calculate Modality Usage Metrics ---
    total_modality_requests = sum(modality_request_counts.values())
    modality_usage = {
        'total_modality_requests': total_modality_requests,
        'average_modalities_per_patient': total_modality_requests / total_patients if total_patients > 0 else 0,
        'request_counts': modality_request_counts,
        'percentage_of_patients_requesting_modality': {
            modality: count / total_patients if total_patients > 0 else 0
            for modality, count in modality_request_counts.items()
        },
        'percentage_request_when_available': {
            'cxr': modality_request_counts['cxr'] / modality_availability_counts['cxr'] if modality_availability_counts['cxr'] > 0 else 0,
            'radiology_report': modality_request_counts['radiology_report'] / modality_availability_counts['radiology_report'] if modality_availability_counts['radiology_report'] > 0 else 0,
        }
    }

    # --- 4. Combine all metrics into a final dictionary ---
    final_metrics = {
        "in_hospital_mortality_metrics": mortality_metrics,
        "modality_usage_analysis": modality_usage,
        "total_patients_evaluated": total_patients
    }

    return final_metrics