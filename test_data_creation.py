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
# 2. HELPER FUNCTIONS
# =====================================================================================
def parse_discharge_summary(note: str):
    if not isinstance(note, str):
        return {key: "" for key in ['hpi', 'pmh', 'psh', 'social_history', 'family_history', 'meds_on_admission']}
    patterns = {
        'hpi': re.compile(r'History of Present Illness:([\s\S]*?)(?=\n\s*\n|[A-Z][a-zA-Z\s]+:)', re.IGNORECASE),
        'pmh': re.compile(r'(?:Past Medical History|PMH):([\s\S]*?)(?=\n\s*\n|[A-Z][a-zA-Z\s]+:)', re.IGNORECASE),
        'psh': re.compile(r'(?:Past Surgical History|PSH):([\s\S]*?)(?=\n\s*\n|[A-Z][a-zA-Z\s]+:)', re.IGNORECASE),
        'social_history': re.compile(r'Social History:([\s\S]*?)(?=\n\s*\n|[A-Z][a-zA-Z\s]+:)', re.IGNORECASE),
        'family_history': re.compile(r'Family History:([\s\S]*?)(?=\n\s*\n|[A-Z][a-zA-Z\s]+:)', re.IGNORECASE),
        'meds_on_admission': re.compile(r'Medications on Admission:([\s\S]*?)(?=\n\s*\n|[A-Z][a-zA-Z\s]+:)', re.IGNORECASE)
    }
    extracted = {}
    for key, pattern in patterns.items():
        match = pattern.search(note)
        extracted[key] = match.group(1).strip() if match else "Not available"
    return extracted


def create_patient_summary(row: pd.Series, parsed_notes: dict) -> str:
    return (
        f"PATIENT BASELINE SUMMARY:\n------------------------\n"
        f"Demographics:\n - Age: {row.get('age', 'N/A')}\n - Gender: {row.get('gender', 'N/A')}\n - Ethnicity: {row.get('ethnicity', 'N/A')}\n\n"
        f"History of Present Illness:\n{parsed_notes['hpi']}\n\n"
        f"Past Medical History:\n{parsed_notes['pmh']}\n\n"
        f"Past Surgical History:\n{parsed_notes['psh']}\n\n"
        f"Social History:\n{parsed_notes['social_history']}\n\n"
        f"Family History:\n{parsed_notes['family_history']}\n\n"
        f"Medications on Admission:\n{parsed_notes['meds_on_admission']}\n"
    )

# =====================================================================================
# 3. CORE PIPELINE (DATAFUSION-ALIGNED)
# =====================================================================================
def create_dataset(args):
    print("Step 1: Loading raw data files... 📂")
    icu_stays = pd.read_csv(os.path.join(args.mimic_iv_dir, 'all_stays.csv'))
    cxr_meta = pd.read_csv(os.path.join(args.mimic_cxr_dir, 'mimic-cxr-2.0.0-metadata.csv'))
    discharge = pd.read_csv(os.path.join(args.mimic_notes_dir, 'discharge.csv'))
    radiology = pd.read_csv(os.path.join(args.mimic_notes_dir, 'radiology.csv'))

    print(f"Loaded ICU stays: {len(icu_stays)}, CXR meta: {len(cxr_meta)}, discharge: {len(discharge)}, radiology: {len(radiology)}")

    print("Step 2: Loading EHR labels... 🏷️")
    pheno_cols = ['stay_id'] + PHENOTYPING_CLASSES
    pheno_df = pd.read_csv(os.path.join(args.ehr_data_dir, 'phenotyping/test_listfile.csv'), usecols=pheno_cols)
    mort_df = pd.read_csv(os.path.join(args.ehr_data_dir, 'in-hospital-mortality/test_listfile.csv'), usecols=['stay_id', 'y_true'])
    mort_df.rename(columns={'y_true': 'in_hospital_mortality_48hr'}, inplace=True)
    master_labels_df = pd.merge(mort_df, pheno_df, on='stay_id', how='left')

    # Add hadm_id, subject_id, and other core info to the master labels df, which will be our base
    final_df = master_labels_df.merge(
        icu_stays[['stay_id', 'hadm_id', 'subject_id', 'intime', 'outtime', 'age', 'gender', 'ethnicity']],
        on='stay_id',
        how='left'
    )
    final_df['intime'] = pd.to_datetime(final_df['intime'])
    print(f"Labels merged with ICU stay info: {len(final_df)} unique stays")

    # ==========================================================
    # STEP 3-5: SEQUENTIAL MERGE & FILTER (Corrected Logic)
    # ==========================================================
    print("\nStep 3: Processing and merging modalities sequentially...")

    # --- Process and Merge CXR data ---
    print("  - Merging CXR data...")
    cxr_meta['StudyTime'] = cxr_meta['StudyTime'].apply(lambda x: f'{int(float(x)):06}' if pd.notna(x) else None)
    cxr_meta['StudyDateTime'] = pd.to_datetime(
        cxr_meta['StudyDate'].astype(str) + ' ' + cxr_meta['StudyTime'].astype(str),
        format='%Y%m%d %H%M%S', errors='coerce'
    )
    cxr_merged = final_df.merge(
        cxr_meta[['subject_id', 'dicom_id', 'StudyDateTime', 'ViewPosition']],
        on=['subject_id'],
        how='left'
    )
    
    print("  - Filtering CXRs to first 48 hours and AP view...")
    end_time_48hr = cxr_merged['intime'] + timedelta(hours=48)
    cxr_48 = cxr_merged[
        (cxr_merged['StudyDateTime'] >= cxr_merged['intime']) &
        (cxr_merged['StudyDateTime'] <= end_time_48hr) &
        (cxr_merged['ViewPosition'].str.upper() == 'AP')
    ]
    cxr_deduplicated = cxr_48.sort_values('StudyDateTime', ascending=False).drop_duplicates('stay_id')
    final_df = final_df.merge(
        cxr_deduplicated[['stay_id', 'dicom_id']],
        on='stay_id',
        how='left'
    )

    # --- Process and Merge Radiology Reports ---
    print("  - Merging Radiology Reports...")
    radiology = radiology.rename(columns={'charttime': 'radiology_charttime', 'text': 'radiology_text'})
    radiology['radiology_charttime'] = pd.to_datetime(radiology['radiology_charttime'], errors='coerce')
    rr_merged = final_df.merge(
        radiology[['subject_id', 'hadm_id', 'radiology_charttime', 'radiology_text']],
        on=['subject_id', 'hadm_id'],
        how='left'
    )

    print("  - Filtering Radiology Reports to first 48 hours and aggregating...")
    end_time_48hr_rr = rr_merged['intime'] + timedelta(hours=48)
    rr_48 = rr_merged[
        (rr_merged['radiology_charttime'] >= rr_merged['intime']) &
        (rr_merged['radiology_charttime'] <= end_time_48hr_rr)
    ]
    rr_48['radiology_text'] = rr_48['radiology_text'].fillna('').astype(str)
    rr_agg = rr_48.groupby('stay_id', as_index=False).agg(radiology_text_agg=('radiology_text', ' '.join))
    final_df = final_df.merge(rr_agg, on='stay_id', how='left')
    final_df.rename(columns={'radiology_text_agg': 'radiology_text'}, inplace=True)


    # --- Process and Merge Discharge Notes ---
    print("  - Merging Discharge Notes...")
    discharge = discharge.rename(columns={'text': 'discharge_text'})
    final_df = final_df.merge(
        discharge[['subject_id', 'hadm_id', 'discharge_text']].drop_duplicates(subset=['hadm_id']),
        on=['subject_id', 'hadm_id'],
        how='left'
    )
    print(f"Final merged multimodal dataset shape: {final_df.shape}")

    # ==========================================================
    # STEP 6: MODALITY STATS BEFORE FILTERING
    # ==========================================================
    print("\n📊 Modality availability BEFORE dropping missing discharge notes:")
    total = len(final_df)
    if total == 0:
        print("⚠️ No samples found before filtering. Exiting.")
        return
    cxr_count = final_df['dicom_id'].notna().sum()
    rr_count = final_df['radiology_text'].notna().sum()
    dn_count = final_df['discharge_text'].notna().sum()
    print(f"Total samples: {total}")
    print(f"  - 🩺 EHR (timeseries): {total}/{total} (100.00%)")
    print(f"  - 🖼️ CXR (AP view): {cxr_count}/{total} ({cxr_count/total:.2%})")
    print(f"  - 📄 Radiology: {rr_count}/{total} ({rr_count/total:.2%})")
    print(f"  - 📜 Discharge: {dn_count}/{total} ({dn_count/total:.2%})")

    # ==========================================================
    # STEP 7: DROP MISSING DISCHARGE NOTES
    # ==========================================================
    print("\nStep 7: Dropping rows with missing discharge_text... 🗑️")
    before = len(final_df)
    final_df = final_df.dropna(subset=['discharge_text'])
    after = len(final_df)
    print(f"Removed {before - after} rows. Remaining: {after}")

    # ==========================================================
    # STEP 8: MODALITY STATS AFTER FILTERING
    # ==========================================================
    if after == 0:
        print("⚠️ No samples remain after discharge filtering. Exiting.")
        return

    cxr_after = final_df['dicom_id'].notna().sum()
    rr_after = final_df['radiology_text'].notna().sum()
    print("\n📊 Modality availability AFTER filtering:")
    print(f"  - 🖼️ CXR (AP view): {cxr_after}/{after} ({cxr_after/after:.2%})")
    print(f"  - 📄 Radiology: {rr_after}/{after} ({rr_after/after:.2%})")
    print(f"  - 📜 Discharge: {after}/{after} (100.00%)")
    
    
    # ==========================================================
    # STEP 8.5: CALCULATE AND DISPLAY LABEL DISTRIBUTIONS
    # ==========================================================
    print("\n📈 Final Label Distributions:")
    total_samples = len(final_df)
    all_label_cols = ['in_hospital_mortality_48hr'] + PHENOTYPING_CLASSES
    label_counts = final_df[all_label_cols].sum()
    
    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  - {label:<55} | Positive cases: {int(count):>4}/{total_samples} ({percentage:5.2f}%)")

    # ==========================================================
    # STEP 9: OUTPUT JSONL
    # ==========================================================
    print(f"\nStep 9: Writing final JSONL output ({after} stays)... ✍️")
    with open(args.output_jsonl, 'w') as f:
        for _, row in tqdm(final_df.iterrows(), total=after, desc="Writing records"):
            parsed_notes = parse_discharge_summary(row['discharge_text'])
            patient_summary = create_patient_summary(row, parsed_notes)

            cxr_path = os.path.join(args.cxr_image_dir, f"{row['dicom_id']}.jpg") if pd.notna(row.get('dicom_id')) else "CXR not available"
            radiology_report = row['radiology_text'] if pd.notna(row['radiology_text']) else "Radiology report not available"
            labels = {lbl: int(row[lbl]) for lbl in PHENOTYPING_CLASSES}
            labels['in_hospital_mortality_48hr'] = int(row['in_hospital_mortality_48hr'])

            ehr_path = os.path.join(args.ehr_data_dir, 'in-hospital-mortality', 'train', f"{row['stay_id']}.csv")
            record = {
                'stay_id': row['stay_id'],
                'hadm_id': int(row['hadm_id']) if pd.notna(row['hadm_id']) else None,
                'patient_summary_text': patient_summary,
                'ehr_timeseries_path': ehr_path,
                'cxr_image_path': cxr_path,
                'radiology_report_text': radiology_report,
                'labels': labels
            }
            f.write(json.dumps(record) + '\n')

    print(f"\n✅ Success! Final multimodal dataset written to {args.output_jsonl}")

# =====================================================================================
# 4. EXECUTION
# =====================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates a multimodal MIMIC-IV dataset (DataFusion-aligned).")
    parser.add_argument('--mimic_iv_dir', type=str, required=True)
    parser.add_argument('--mimic_cxr_dir', type=str, required=True)
    parser.add_argument('--mimic_notes_dir', type=str, required=True)
    parser.add_argument('--ehr_data_dir', type=str, required=True)
    parser.add_argument('--cxr_image_dir', type=str, required=True)
    parser.add_argument('--output_jsonl', type=str, required=True)
    args = parser.parse_args()

    create_dataset(args)