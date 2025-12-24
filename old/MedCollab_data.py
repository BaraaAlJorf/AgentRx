import pandas as pd
import numpy as np
import re
import os
import json
import argparse
from tqdm import tqdm
from datetime import timedelta

# =====================================================================================
# 1. CONSTANTS
# =====================================================================================
PHENOTYPING_CLASSES = [
    'Acute and unspecified renal failure', 'Acute cerebrovascular disease',
    'Acute myocardial infarction', 'Cardiac dysrhythmias', 'Chronic kidney disease',
    'Chronic obstructive pulmonary disease and bronchiectasis',
    'Complications of surgical procedures or medical care', 'Conduction disorders',
    'Congestive heart failure; nonhypertensive', 'Coronary atherosclerosis and other heart disease',
    'Diabetes mellitus with complications', 'Diabetes mellitus without complication',
    'Disorders of lipid metabolism', 'Essential hypertension', 'Fluid and electrolyte disorders',
    'Gastrointestinal hemorrhage', 'Hypertension with complications and secondary hypertension',
    'Other liver diseases', 'Other lower respiratory disease', 'Other upper respiratory disease',
    'Pleurisy; pneumothorax; pulmonary collapse',
    'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
    'Respiratory failure; insufficiency; arrest (adult)', 'Septicemia (except in labor)', 'Shock'
]

# =====================================================================================
# 2. HELPER FUNCTIONS for Text Parsing
# =====================================================================================
def parse_discharge_summary(note: str):
    """Parses a discharge summary to extract key medical history sections."""
    if not isinstance(note, str): return {key: "" for key in ['hpi', 'pmh', 'psh', 'social_history', 'family_history', 'meds_on_admission']}
    patterns = {
        'hpi': re.compile(r'History of Present Illness:([\s\S]*?)(?=\n\s*\n|[A-Z][a-zA-Z\s]+:)', re.IGNORECASE),
        'pmh': re.compile(r'(?:Past Medical History|PMH):([\s\S]*?)(?=\n\s*\n|[A-Z][a-zA-Z\s]+:)', re.IGNORECASE),
        'psh': re.compile(r'(?:Past Surgical History|PSH):([\s\S]*?)(?=\n\s*\n|[A-Z][a-zA-Z\s]+:)', re.IGNORECASE),
        'social_history': re.compile(r'Social History:([\s\S]*?)(?=\n\s*\n|[A-Z][a-zA-Z\s]+:)', re.IGNORECASE),
        'family_history': re.compile(r'Family History:([\s\S]*?)(?=\n\s*\n|[A-Z][a-zA-Z\s]+:)', re.IGNORECASE),
        'meds_on_admission': re.compile(r'Medications on Admission:([\s\S]*?)(?=\n\s*\n|[A-Z][a-zA-Z\s]+:)', re.IGNORECASE)
    }
    extracted_sections = {}
    for key, pattern in patterns.items():
        match = pattern.search(note)
        extracted_sections[key] = match.group(1).strip() if match else "Not available"
    return extracted_sections

def create_patient_summary(row: pd.Series, parsed_notes: dict) -> str:
    """Creates a single text block summarizing the patient's baseline information."""
    summary = (
        f"PATIENT BASELINE SUMMARY:\n------------------------\n"
        f"Demographics:\n - Age: {row.get('age', 'N/A')}\n - Gender: {row.get('gender', 'N/A')}\n - Ethnicity: {row.get('ethnicity', 'N/A')}\n\n"
        f"History of Present Illness:\n{parsed_notes['hpi']}\n\n"
        f"Past Medical History:\n{parsed_notes['pmh']}\n\n"
        f"Past Surgical History:\n{parsed_notes['psh']}\n\n"
        f"Social History:\n{parsed_notes['social_history']}\n\n"
        f"Family History:\n{parsed_notes['family_history']}\n\n"
        f"Medications on Admission:\n{parsed_notes['meds_on_admission']}\n"
    )
    return summary

# =====================================================================================
# 3. CORE DATA FUSION AND PROCESSING
# =====================================================================================
def create_dataset(args):
    """Main function to perform all data loading, merging, and inclusive processing."""
    
    print("Step 1: Loading all raw data files... 📂")
    icu_stays = pd.read_csv(os.path.join(args.mimic_iv_dir, 'all_stays.csv'))
    cxr_meta = pd.read_csv(os.path.join(args.mimic_cxr_dir, 'mimic-cxr-2.0.0-metadata.csv'))
    discharge = pd.read_csv(os.path.join(args.mimic_notes_dir, 'discharge.csv'))
    radiology = pd.read_csv(os.path.join(args.mimic_notes_dir, 'radiology.csv'))

    # Corrected and Simplified Code Block
    print("Step 2: Loading labels, centered on stays with mortality data... 🏷️")
    
    # --- Process Phenotyping Labels ---
    # Define the exact columns to load, using 'stay_id' and ignoring the 'stay' column.
    pheno_cols_to_load = ['stay_id'] + PHENOTYPING_CLASSES
    #pheno_splits = [pd.read_csv(os.path.join(args.ehr_data_dir, 'phenotyping', f'{s}_listfile.csv'), usecols=pheno_cols_to_load) for s in ['train', 'val', 'test']]
    pheno_splits = [pd.read_csv(os.path.join(args.ehr_data_dir, 'phenotyping', f'{s}_listfile.csv'), usecols=pheno_cols_to_load) for s in ['test']]
    pheno_labels_df = pd.concat(pheno_splits)
    
    # Ensure phenotyping labels have unique rows
    pheno_labels_df.drop_duplicates(subset=['stay_id'], keep='first', inplace=True)
    
    
    # --- Process Mortality Labels ---
    # Define the exact columns to load, using 'stay_id' and ignoring the 'stay' column.
    mortality_cols_to_load = ['stay_id', 'y_true']
    #mortality_splits = [pd.read_csv(os.path.join(args.ehr_data_dir, 'in-hospital-mortality', f'{s}_listfile.csv'), usecols=mortality_cols_to_load) for s in ['train', 'val', 'test']]
    mortality_splits = [pd.read_csv(os.path.join(args.ehr_data_dir, 'in-hospital-mortality', f'{s}_listfile.csv'), usecols=mortality_cols_to_load) for s in ['test']]
    mortality_labels_df = pd.concat(mortality_splits)
    
    # Ensure mortality labels have unique rows
    mortality_labels_df.drop_duplicates(subset=['stay_id'], keep='first', inplace=True)
    mortality_labels_df = mortality_labels_df.rename(columns={'y_true': 'in_hospital_mortality_48hr'})
    
    # Now the merge will work correctly
    master_labels_df = pd.merge(mortality_labels_df, pheno_labels_df, on='stay_id', how='left')

    print("Step 3: Merging all clinical data sources with left joins... 🔄")
    clinical_df = icu_stays.merge(cxr_meta, on='subject_id', how='left')
    clinical_df = clinical_df.merge(discharge[['subject_id', 'hadm_id', 'charttime', 'text']], on=['subject_id', 'hadm_id'], how='left')
    clinical_df.rename(columns={'text': 'discharge_text', 'charttime': 'discharge_charttime'}, inplace=True)
    clinical_df = clinical_df.merge(radiology[['subject_id', 'hadm_id', 'charttime', 'text']], on=['subject_id', 'hadm_id'], how='left')
    clinical_df.rename(columns={'text': 'radiology_text', 'charttime': 'radiology_charttime'}, inplace=True)

    print("Step 4: Filtering modalities to first 48 hours of ICU stay... ⏳")
    for col in ['intime', 'outtime', 'StudyDate', 'discharge_charttime', 'radiology_charttime']:
        clinical_df[col] = pd.to_datetime(clinical_df[col], errors='coerce')
    clinical_df['StudyTime_str'] = clinical_df['StudyTime'].apply(lambda x: f'{int(float(x)):06d}' if pd.notna(x) else None)
    clinical_df['StudyDateTime'] = pd.to_datetime(clinical_df['StudyDate'].dt.strftime('%Y%m%d') + ' ' + clinical_df['StudyTime_str'], format='%Y%m%d %H%M%S', errors='coerce')
    end_time_48hr = clinical_df['intime'] + timedelta(hours=48)
    
    # Create a dataframe of all valid events within the 48 hour window
    valid_events_df = clinical_df[clinical_df['intime'].notna()].copy()
    valid_events_df = valid_events_df[valid_events_df['StudyDateTime'].between(valid_events_df['intime'], end_time_48hr) | valid_events_df['radiology_charttime'].between(valid_events_df['intime'], end_time_48hr)]

    # --- 3.5 HYBRID AGGREGATION LOGIC ---
    print("Step 5: Aggregating reports and selecting latest CXR per stay... 🎯")
    # 5a. Aggregate all unique radiology reports in the window
    rad_agg = valid_events_df.dropna(subset=['radiology_text', 'radiology_charttime'])
    rad_agg = rad_agg.sort_values('radiology_charttime').groupby('stay_id')['radiology_text'].apply(lambda texts: '\n\n--- END OF REPORT ---\n\n'.join(texts.unique())).reset_index()

    # 5b. Select the single latest CXR and its associated metadata in the window
    cxr_latest = valid_events_df.dropna(subset=['dicom_id', 'StudyDateTime'])
    cxr_latest = cxr_latest[cxr_latest['ViewPosition'] == 'AP']
    cxr_latest = cxr_latest.sort_values('StudyDateTime').groupby('stay_id').tail(1)
    
    # Keep only necessary columns to avoid data duplication from other merges
    patient_info_cols = ['stay_id', 'hadm_id', 'subject_id', 'age', 'gender', 'ethnicity', 'dicom_id', 'discharge_text']
    cxr_latest = cxr_latest[patient_info_cols]

    # 5c. Combine master labels with the processed clinical data
    final_df = master_labels_df.merge(cxr_latest, on='stay_id', how='left')
    final_df = final_df.merge(rad_agg, on='stay_id', how='left')
    final_df[PHENOTYPING_CLASSES] = final_df[PHENOTYPING_CLASSES].fillna(0).astype(int)
    final_df['in_hospital_mortality_48hr'] = final_df['in_hospital_mortality_48hr'].astype(int)

    # --- 3.6 Generate Final JSONL Output ---
    print(f"Step 6: Generating final JSONL file for {len(final_df)} stays... ✍️")
    with open(args.output_jsonl, 'w') as f:
        for _, row in tqdm(final_df.iterrows(), total=final_df.shape[0], desc="Writing records"):
            if pd.notna(row.get('discharge_text')):
                parsed_notes = parse_discharge_summary(row['discharge_text'])
                patient_summary = create_patient_summary(row, parsed_notes)
            else:
                patient_summary = "Patient summary not yet available (no discharge note found)."
            
            cxr_path = os.path.join(args.cxr_image_dir, f"{row['dicom_id']}.jpg") if pd.notna(row.get('dicom_id')) else "CXR image not yet available (no valid image in first 48 hours)."
            radiology_report = row['radiology_text'] if pd.notna(row.get('radiology_text')) and row['radiology_text'].strip() else "Radiology report not yet available (no report in first 48 hours)."
            
            labels = {label: int(row[label]) for label in PHENOTYPING_CLASSES}
            labels['in_hospital_mortality_48hr'] = int(row['in_hospital_mortality_48hr'])
            ehr_path = os.path.join(args.ehr_data_dir, 'in-hospital-mortality', 'train', f"{row['stay_id']}.csv")

            record = {
                'stay_id': row['stay_id'],
                'hadm_id': int(row['hadm_id']) if pd.notna(row.get('hadm_id')) else None,
                'patient_summary_text': patient_summary,
                'ehr_timeseries_path': ehr_path,
                'cxr_image_path': cxr_path,
                'radiology_report_text': radiology_report,
                'labels': labels
            }
            f.write(json.dumps(record) + '\n')
            
    print(f"\n✅ Success! Inclusive dataset created at: {args.output_jsonl}")

# =====================================================================================
# 4. SCRIPT EXECUTION
# =====================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates an inclusive multimodal dataset, pairing the latest CXR with all aggregated radiology reports from the first 48 hours.")
    
    parser.add_argument('--mimic_iv_dir', type=str, required=True, help="Path to MIMIC-IV CSVs (e.g., all_stays.csv).")
    parser.add_argument('--mimic_cxr_dir', type=str, required=True, help="Path to MIMIC-CXR CSVs (e.g., mimic-cxr-2.0.0-metadata.csv).")
    parser.add_argument('--mimic_notes_dir', type=str, required=True, help="Path to MIMIC-IV-Note CSVs (e.g., discharge.csv, radiology.csv).")
    parser.add_argument('--ehr_data_dir', type=str, required=True, help="Path to the root of your prepared EHR data (containing 'phenotyping' and 'in-hospital-mortality' subfolders).")
    parser.add_argument('--cxr_image_dir', type=str, required=True, help="Path to the directory containing resized CXR jpg images.")
    parser.add_argument('--output_jsonl', type=str, required=True, help="Full path for the final output .jsonl file.")
    
    args = parser.parse_args()
    create_dataset(args)