#!/usr/bin/env python3
import os
import argparse
import pandas as pd

# ----------------------------
# Helper loaders (adapted from your pipeline)
# ----------------------------
def load_ehr_benchmark_data(ehr_data_dir: str) -> pd.DataFrame:
    task_path = os.path.join(ehr_data_dir, 'in-hospital-mortality')
    all_dfs = []
    for split in ['train', 'val', 'test']:
        listfile_path = os.path.join(task_path, f'{split}_listfile.csv')
        if not os.path.exists(listfile_path):
            continue
        df = pd.read_csv(listfile_path)
        filename_col = df.columns[0]
        stay_id_col = next((col for col in df.columns if 'stay' in col.lower() and 'id' in col.lower()), df.columns[2])
        out = pd.DataFrame()
        out['stay_id'] = df[stay_id_col].astype(int)
        out['split'] = split
        out['ehr_timeseries_path'] = df[filename_col].apply(lambda x: os.path.join(task_path, 'train' if split=='val' else split, str(x)))
        all_dfs.append(out)
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def load_all_stays(mimic_iv_dir: str) -> pd.DataFrame:
    path = os.path.join(mimic_iv_dir, 'all_stays.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df['stay_id'] = df['stay_id'].astype(int)
    # keep hadm_id and subject_id as ints if possible
    df['subject_id'] = df['subject_id'].astype(int)
    # hadm_id can be nullable in some dumps, handle carefully
    if 'hadm_id' in df.columns:
        df['hadm_id'] = df['hadm_id'].astype(pd.Int64Dtype())
    return df

def load_downstream_parquet(parquet_path: str) -> pd.DataFrame:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)
    df = pd.read_parquet(parquet_path)
    return df

# ----------------------------
# Core compare logic
# ----------------------------
def compare_mappings(ehr_data_dir, mimic_iv_dir, downstream_parquet, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1) load benchmark stays (stay_id)
    master_df = load_ehr_benchmark_data(ehr_data_dir)
    if master_df.empty:
        raise RuntimeError("No benchmark listfiles loaded.")
    print(f"[1] Loaded {len(master_df)} benchmark stay rows (unique stays: {master_df['stay_id'].nunique()})")

    # 2) attach icu metadata to get subject_id & hadm_id
    icu_stays = load_all_stays(mimic_iv_dir)
    merged = master_df.merge(
        icu_stays[['stay_id', 'hadm_id', 'subject_id', 'intime']],
        on='stay_id', how='left', indicator=True
    )
    print(f"[2] After merge with all_stays: {len(merged)} rows; merge indicator counts:")
    print(merged['_merge'].value_counts().to_string())
    # Save stays that didn't find a match in all_stays
    merged[merged['_merge'] == 'left_only'][['stay_id']].drop_duplicates().to_csv(os.path.join(output_dir, 'stays_not_in_all_stays.csv'), index=False)

    # 3) load downstream parquet
    downstream = load_downstream_parquet(downstream_parquet)
    print(f"[3] Downstream parquet columns: {list(downstream.columns)}")

    # -------------------------
    # A: Stay-based labels
    # -------------------------
    # find stay id column in downstream: 'stay_id' or 'icustay_id' or 'icustay_id'
    stay_col = None
    for candidate in ['stay_id', 'icustay_id', 'icustayid', 'icustayId', 'icustay_ID']:
        if candidate in downstream.columns:
            stay_col = candidate
            break

    stay_labels_df = None
    if stay_col is not None:
        stay_labels_df = downstream[[stay_col, 'y_los_7']].rename(columns={stay_col: 'stay_id'}).copy()
        stay_labels_df['stay_id'] = stay_labels_df['stay_id'].astype(int)
        stay_labels_df['y_los_7'] = stay_labels_df['y_los_7'].astype(int)
        # drop duplicates keep first
        stay_labels_df = stay_labels_df.drop_duplicates(subset=['stay_id'], keep='first').reset_index(drop=True)
        print(f"[A] Found stay-based label column '{stay_col}' with {len(stay_labels_df)} unique stay labels.")
    else:
        print("[A] No stay-based column found in downstream parquet. Stay-based comparison will be skipped.")

    # Merge master stays with stay-based labels (on stay_id)
    if stay_labels_df is not None:
        merged_staylabel = merged.merge(stay_labels_df, on='stay_id', how='left', suffixes=('','_stay'))
        # which benchmark stays had a stay-level label available
        stay_labeled = merged_staylabel[merged_staylabel['y_los_7'].notna()]
        print(f"[A] Stays matched to stay-based labels: {len(stay_labeled)} (of {len(merged)})")
        stay_labeled[['stay_id','subject_id','hadm_id','split','ehr_timeseries_path','y_los_7']].to_csv(os.path.join(output_dir, 'merged_stay_labels.csv'), index=False)
    else:
        merged_staylabel = merged.copy()
        merged_staylabel['y_los_7'] = pd.NA

    # -------------------------
    # B: Admission-based labels (subject_id + hadm_id)
    # -------------------------
    # check downstream has subject_id + hadm_id
    if {'subject_id','hadm_id','y_los_7'}.issubset(set(downstream.columns)):
        adm_labels_df = downstream[['subject_id','hadm_id','y_los_7']].copy()
        adm_labels_df['subject_id'] = adm_labels_df['subject_id'].astype(int)
        adm_labels_df['hadm_id'] = adm_labels_df['hadm_id'].astype(int)
        adm_labels_df['y_los_7'] = adm_labels_df['y_los_7'].astype(int)
        adm_labels_df = adm_labels_df.drop_duplicates(subset=['subject_id','hadm_id'], keep='first').reset_index(drop=True)
        print(f"[B] Admission-based labels (subject+hadm) rows: {len(adm_labels_df)}")
    else:
        raise RuntimeError("downstream parquet missing required columns for admission-based labels: 'subject_id', 'hadm_id', 'y_los_7'")

    # Merge master stays (which have subject_id & hadm_id from all_stays) with admission labels
    merged_admlabel = merged.merge(adm_labels_df, on=['subject_id','hadm_id'], how='left', suffixes=('','_adm'))
    adm_labeled = merged_admlabel[merged_admlabel['y_los_7'].notna()]
    print(f"[B] Stays matched to admission-based labels: {len(adm_labeled)} (of {len(merged)})")
    adm_labeled[['stay_id','subject_id','hadm_id','split','ehr_timeseries_path','y_los_7']].to_csv(os.path.join(output_dir, 'merged_adm_labels.csv'), index=False)

    # -------------------------
    # Compare sets
    # -------------------------
    # sets of stay ids labeled by each method
    stays_staylabel_set = set(merged_staylabel[merged_staylabel['y_los_7'].notna()]['stay_id'].unique())
    stays_admlabel_set = set(merged_admlabel[merged_admlabel['y_los_7'].notna()]['stay_id'].unique())

    in_both = stays_staylabel_set & stays_admlabel_set
    only_stay = stays_staylabel_set - stays_admlabel_set
    only_adm = stays_admlabel_set - stays_staylabel_set

    print("\n=== OVERLAP SUMMARY ===")
    print(f"Total benchmark stays: {len(merged)}")
    print(f"Stays with stay-based label: {len(stays_staylabel_set)}")
    print(f"Stays with adm-based label : {len(stays_admlabel_set)}")
    print(f"Intersection (both):       {len(in_both)}")
    print(f"Only stay-based:           {len(only_stay)}")
    print(f"Only adm-based:            {len(only_adm)}")
    if len(merged) > 0:
        print(f"Fraction matched by at least one method: {(len(stays_staylabel_set | stays_admlabel_set) / len(merged)):.2%}")

    # write CSVs for inspection
    pd.DataFrame(sorted(list(only_stay)), columns=['stay_id']).to_csv(os.path.join(output_dir,'only_staylabel_stays.csv'), index=False)
    pd.DataFrame(sorted(list(only_adm)), columns=['stay_id']).to_csv(os.path.join(output_dir,'only_admlabel_stays.csv'), index=False)
    pd.DataFrame(sorted(list(in_both)), columns=['stay_id']).to_csv(os.path.join(output_dir,'bothlabel_stays.csv'), index=False)

    # -------------------------
    # Check label disagreements where both present
    # -------------------------
    if len(in_both) > 0:
        # build dataframes keyed by stay_id with label columns for each approach
        stay_side = merged_staylabel[['stay_id','y_los_7']].dropna().drop_duplicates(subset=['stay_id']).set_index('stay_id')
        adm_side = merged_admlabel[['stay_id','y_los_7']].dropna().drop_duplicates(subset=['stay_id']).set_index('stay_id')
        compare_df = stay_side.join(adm_side, how='inner', lsuffix='_stay', rsuffix='_adm').reset_index()
        compare_df.columns = ['stay_id','y_los_7_stay','y_los_7_adm']
        compare_df['agree'] = compare_df['y_los_7_stay'] == compare_df['y_los_7_adm']
        disagreements = compare_df[compare_df['agree'] == False]
        print(f"\nLabel agreement in intersection: {compare_df['agree'].mean():.2%} agree ({len(disagreements)} disagreed out of {len(compare_df)})")
        disagreements.to_csv(os.path.join(output_dir,'label_disagreements.csv'), index=False)
        # show some examples
        if not disagreements.empty:
            print("\nExamples of disagreements (first 10):")
            print(disagreements.head(10).to_string(index=False))
    else:
        print("\nNo intersection between methods to compare labels on the same stay_id.")

    print(f"\nDiagnostics written to {output_dir} (merged_stay_labels.csv, merged_adm_labels.csv, only_staylabel_stays.csv, only_admlabel_stays.csv, bothlabel_stays.csv, label_disagreements.csv)")

# ----------------------------
# CLI
# ----------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ehr_data_dir', required=True)
    p.add_argument('--mimic_iv_dir', required=True)
    p.add_argument('--downstream_parquet', required=True)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()
    compare_mappings(args.ehr_data_dir, args.mimic_iv_dir, args.downstream_parquet, args.output_dir)
