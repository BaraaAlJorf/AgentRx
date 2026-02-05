#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import os
import json
import argparse
from tqdm import tqdm
from datetime import timedelta

# =====================================================================================
# 1. HELPERS (mostly unchanged)
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
# 2. Load EHR benchmark paths (labels ignored here)
# =====================================================================================
def load_ehr_benchmark_data(ehr_data_dir: str) -> pd.DataFrame:
    """
    Loads EHR timeseries paths and split information from benchmark listfiles.
    Returns a DataFrame with columns: stay_id, split, ehr_timeseries_path
    """
    print("\nStep 1: Loading EHR Benchmark Labels & Paths (paths only)...")
    task_path = os.path.join(ehr_data_dir, 'in-hospital-mortality')
    all_dfs = []

    for split in ['train', 'val', 'test']:
        listfile_path = os.path.join(task_path, f'{split}_listfile.csv')
        if not os.path.exists(listfile_path):
            print(f"  ⚠️ Warning: {split} listfile not found at {listfile_path}")
            continue

        # Validation files are usually in the 'train' directory
        folder_name = 'train' if split == 'val' else split
        data_dir_path = os.path.join(task_path, folder_name)

        try:
            df = pd.read_csv(listfile_path)

            # 1. Filename is ALWAYS column 0 in these benchmarks
            filename_col = df.columns[0]

            # 2. Stay ID: Look for 'stay_id' or fall back to column 2
            stay_id_col = next((col for col in df.columns if 'stay' in col.lower() and 'id' in col.lower()), df.columns[2])

            out = pd.DataFrame()
            out['stay_id'] = df[stay_id_col].astype(int)
            out['split'] = split
            out['ehr_timeseries_path'] = df[filename_col].apply(lambda x: os.path.join(data_dir_path, str(x)))

            all_dfs.append(out)

        except Exception as e:
            print(f"  ❌ Error reading {split} file: {e}")
            continue

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

# =====================================================================================
# 3. Load downstream labels (from parquet) — will be merged by subject_id & hadm_id later
# =====================================================================================
def load_downstream_labels(parquet_path: str) -> pd.DataFrame:
    """
    Loads labels from downstream_idx.parquet
    Expects columns: subject_id, hadm_id, y_los_7
    """
    print("\nStep 1.5: Loading downstream labels from parquet...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"downstream parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    required = {'subject_id', 'hadm_id', 'y_los_7'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"downstream_idx.parquet is missing required columns: {missing}")

    df = df[['subject_id', 'hadm_id', 'y_los_7']].copy()
    df['subject_id'] = df['subject_id'].astype(int)
    df['hadm_id'] = df['hadm_id'].astype(int)
    # cast label to int (assumption: integer or 0/1)
    df['y_los_7'] = df['y_los_7'].astype(int)

    # Ensure uniqueness per admission (subject_id, hadm_id)
    dup_count = df.duplicated(subset=['subject_id', 'hadm_id']).sum()
    if dup_count > 0:
        print(f"  ⚠️ Warning: {dup_count} duplicate (subject_id, hadm_id) pairs in downstream parquet. Keeping first occurrence.")
        df = df.drop_duplicates(subset=['subject_id', 'hadm_id'], keep='first')

    return df

# =====================================================================================
# 4. Core pipeline: build dataset, merge ICU metadata, join labels by (subject_id, hadm_id)
# =====================================================================================
def create_dataset(args):
    # --- 1. Load EHR paths ---
    master_df = load_ehr_benchmark_data(args.ehr_data_dir)
    if master_df.empty:
        print("❌ Error: No benchmark listfiles found or no rows parsed.")
        return

    # --- 2. Load ICU metadata (to get subject_id & hadm_id for each stay_id) ---
    print("\nStep 2: Loading auxiliary MIMIC metadata... 📂")
    icu_stays_path = os.path.join(args.mimic_iv_dir, 'all_stays.csv')
    if not os.path.exists(icu_stays_path):
        print(f"❌ Error: expected all_stays.csv at {icu_stays_path}")
        return

    icu_stays = pd.read_csv(icu_stays_path)
    icu_stays['stay_id'] = icu_stays['stay_id'].astype(int)

    # Merge to attach admission/subject info to each stay
    print("\nStep 2.5: Merging Benchmark Data with ICU Stays...")
    merged = master_df.merge(
        icu_stays[['stay_id', 'hadm_id', 'subject_id', 'intime', 'outtime', 'age', 'gender', 'ethnicity']],
        on='stay_id',
        how='inner'
    )

    if merged.empty:
        print("❌ Error: No matches between benchmark stays and ICU stays (stay_id).")
        return

    merged['intime'] = pd.to_datetime(merged['intime'])
    merged['subject_id'] = merged['subject_id'].astype(int)
    # hadm_id may have NaNs in some datasets; cast carefully
    merged['hadm_id'] = merged['hadm_id'].astype(pd.Int64Dtype())

    print(f"  - Records matched with ICU data: {len(merged)} (Lost {len(master_df) - len(merged)} records)")

    # --- 3. Load downstream labels and merge by (subject_id, hadm_id) ---
    labels_df = load_downstream_labels(args.downstream_parquet)

    # Ensure hadm_id type alignment: labels hadm_id is int; merged hadm_id is Int64 (nullable) => convert
    # We'll drop rows where hadm_id is NA before join because join key requires hadm_id
    pre_drop_na = len(merged)
    merged = merged.dropna(subset=['hadm_id'])
    if len(merged) < pre_drop_na:
        print(f"  - Dropped {pre_drop_na - len(merged)} rows where hadm_id was missing (cannot join labels).")

    merged['hadm_id'] = merged['hadm_id'].astype(int)

    # Merge labels
    merged = merged.merge(labels_df, on=['subject_id', 'hadm_id'], how='inner')
    print(f"  - Records after label merge: {len(merged)}")

    if merged.empty:
        print("❌ Error: No records after merging downstream labels by (subject_id, hadm_id).")
        return

    # ==========================================================
    # STEP 3: LINKING MODALITIES (CXR, Radiology, Discharge)
    # ==========================================================
    print("\nStep 3: Linking Modalities...")

    # --- A. CXR ---
    cxr_meta = pd.read_csv(os.path.join(args.mimic_cxr_dir, 'mimic-cxr-2.0.0-metadata.csv'))
    # Normalize StudyTime -> zero-padded HHMMSS
    if 'StudyTime' in cxr_meta.columns:
        cxr_meta['StudyTime'] = cxr_meta['StudyTime'].apply(lambda x: f'{int(float(x)):06}' if pd.notna(x) else None)
    else:
        cxr_meta['StudyTime'] = None

    # Create StudyDateTime
    cxr_meta['StudyDate'] = cxr_meta['StudyDate'].astype(str)
    cxr_meta['StudyTime'] = cxr_meta['StudyTime'].astype(str)
    cxr_meta['StudyDateTime'] = pd.to_datetime(
        cxr_meta['StudyDate'].astype(str) + ' ' + cxr_meta['StudyTime'].astype(str),
        format='%Y%m%d %H%M%S', errors='coerce'
    )

    # Merge on subject_id to attach cxr records to admissions
    cxr_merged = merged.merge(cxr_meta[['subject_id', 'dicom_id', 'StudyDateTime', 'ViewPosition']], on=['subject_id'], how='left')

    end_time_48hr = cxr_merged['intime'] + timedelta(hours=48)
    cxr_48 = cxr_merged[
        (cxr_merged['StudyDateTime'] >= cxr_merged['intime']) &
        (cxr_merged['StudyDateTime'] <= end_time_48hr) &
        (cxr_merged['ViewPosition'].notna()) &
        (cxr_merged['ViewPosition'].str.upper() == 'AP')
    ]

    # Keep the most recent AP view within 48h per stay
    cxr_deduplicated = cxr_48.sort_values('StudyDateTime', ascending=False).drop_duplicates('stay_id')
    merged = merged.merge(cxr_deduplicated[['stay_id', 'dicom_id']], on='stay_id', how='left')

    # --- B. Radiology ---
    radiology = pd.read_csv(os.path.join(args.mimic_notes_dir, 'radiology.csv'))
    radiology = radiology.rename(columns={'charttime': 'radiology_charttime', 'text': 'radiology_text'})
    radiology['radiology_charttime'] = pd.to_datetime(radiology['radiology_charttime'], errors='coerce')
    rr_merged = merged.merge(radiology[['subject_id', 'hadm_id', 'radiology_charttime', 'radiology_text']], on=['subject_id', 'hadm_id'], how='left')

    end_time_48hr_rr = rr_merged['intime'] + timedelta(hours=48)
    rr_48 = rr_merged[
        (rr_merged['radiology_charttime'] >= rr_merged['intime']) &
        (rr_merged['radiology_charttime'] <= end_time_48hr_rr)
    ].copy()

    rr_48['radiology_text'] = rr_48['radiology_text'].fillna('').astype(str)
    rr_agg = rr_48.groupby('stay_id', as_index=False).agg(radiology_text_agg=('radiology_text', ' '.join))
    merged = merged.merge(rr_agg, on='stay_id', how='left')
    merged.rename(columns={'radiology_text_agg': 'radiology_text'}, inplace=True)

    # --- C. Discharge Notes ---
    discharge = pd.read_csv(os.path.join(args.mimic_notes_dir, 'discharge.csv'))
    discharge = discharge.rename(columns={'text': 'discharge_text'})
    # deduplicate on hadm_id to avoid multiple discharge notes per admission
    merged = merged.merge(
        discharge[['subject_id', 'hadm_id', 'discharge_text']].drop_duplicates(subset=['hadm_id']),
        on=['subject_id', 'hadm_id'],
        how='left'
    )

    # ==========================================================
    # STEP 3.5: STATISTICS BY SPLIT (BEFORE FILTERING)
    # ==========================================================
    print("\n" + "="*50)
    print("📊 DATASET STATISTICS BY SPLIT (BEFORE FILTERING)")
    print("="*50)

    for split in ['train', 'val', 'test']:
        subset = merged[merged['split'] == split]
        total_split = len(subset)

        if total_split == 0:
            print(f"\n⚠️  {split.upper()}: No records found.")
            continue

        cxr_count = subset['dicom_id'].notna().sum()
        rad_count = subset['radiology_text'].notna().sum()
        note_count = subset['discharge_text'].notna().sum()

        print(f"\n[{split.upper()}] Total Records: {total_split}")
        print(f"  ├── With Discharge Note: {note_count:<5} ({note_count/total_split:6.2%}) -> (Required for Final Dataset)")
        print(f"  ├── With CXR Image:      {cxr_count:<5} ({cxr_count/total_split:6.2%})")
        print(f"  └── With Radiology Rpt:  {rad_count:<5} ({rad_count/total_split:6.2%})")

    print("\n" + "="*50 + "\n")

    # ==========================================================
    # STEP 4: FILTERING - require discharge_text (as before)
    # ==========================================================
    print("Step 4: Dropping rows missing discharge notes... 🗑️")
    total_before = len(merged)
    final_df = merged.dropna(subset=['discharge_text']).copy()
    after_count = len(final_df)
    print(f"  - Dropped {total_before - after_count} rows.")
    print(f"  - FINAL COMBINED DATASET SIZE: {after_count}")

    if final_df.empty:
        print("❌ Final dataset empty after requiring discharge notes.")
        return

    # ==========================================================
    # STEP 5: OUTPUT WRITING (JSONL)
    # ==========================================================
    os.makedirs(args.output_dir, exist_ok=True)
    splits = ['train', 'val', 'test']

    for split in splits:
        split_df = final_df[final_df['split'] == split]
        output_file = os.path.join(args.output_dir, f"{split}.jsonl")

        print(f"\nWriting {split.upper()} set to {output_file}...")
        if len(split_df) == 0:
            print(f"  ⚠️ Warning: No records for {split}.")
            continue

        with open(output_file, 'w') as f:
            for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split}"):
                parsed_notes = parse_discharge_summary(row['discharge_text'])
                patient_summary = create_patient_summary(row, parsed_notes)

                cxr_path = (os.path.join(args.cxr_image_dir, f"{row['dicom_id']}.jpg")
                            if pd.notna(row.get('dicom_id')) else "CXR not available")
                radiology_report = row['radiology_text'] if pd.notna(row['radiology_text']) else "Radiology report not available"
                ehr_path = row['ehr_timeseries_path']

                record = {
                    'subject_id': int(row['subject_id']),
                    'stay_id': int(row['stay_id']),
                    'hadm_id': int(row['hadm_id']) if pd.notna(row['hadm_id']) else None,
                    'patient_summary_text': patient_summary,
                    'ehr_timeseries_path': ehr_path,
                    'cxr_image_path': cxr_path,
                    'radiology_report_text': radiology_report,
                    'labels': {
                        'los_7': int(row['y_los_7'])
                    }
                }
                f.write(json.dumps(record) + '\n')

    print(f"\n✅ Success! All splits written to {args.output_dir}")

# =====================================================================================
# 6. Execution / Argparse
# =====================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates a multimodal MIMIC-IV dataset split by Train/Val/Test using labels from a downstream parquet joined by (subject_id, hadm_id).")
    parser.add_argument('--mimic_iv_dir', type=str, required=True)
    parser.add_argument('--mimic_cxr_dir', type=str, required=True)
    parser.add_argument('--mimic_notes_dir', type=str, required=True)
    parser.add_argument('--ehr_data_dir', type=str, required=True)
    parser.add_argument('--downstream_parquet', type=str, required=True, help="Path to downstream_idx.parquet containing subject_id, hadm_id, y_los_7")
    parser.add_argument('--cxr_image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save train.jsonl, val.jsonl, test.jsonl")

    args = parser.parse_args()
    create_dataset(args)
