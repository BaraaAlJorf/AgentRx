#!/usr/bin/env python3
"""
check_downstream_label_inconsistencies.py

Find (subject_id, hadm_id) groups in a downstream parquet that have
conflicting y_mort_1yr or y_los_7 labels.

Produces CSV reports in --out_dir:
 - inconsistent_y_mort_1yr.csv     : (subject_id,hadm_id) with >1 unique y_mort_1yr values
 - inconsistent_y_los_7.csv         : (subject_id,hadm_id) with >1 unique y_los_7 values
 - inconsistent_any_label.csv       : union of the above
 - rows_in_inconsistent_groups_*.csv: raw rows from the parquet for each inconsistent group
"""
import os
import argparse
import pandas as pd

REQUIRED_COLS = {'subject_id', 'hadm_id'}  # labels are checked conditionally

def load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pd.read_parquet(path)
    return df

def ensure_int_columns(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            try:
                df[c] = df[c].astype(int)
            except Exception:
                # try nullable int then fill if possible
                try:
                    df[c] = df[c].astype(pd.Int64Dtype())
                except Exception:
                    # leave as-is but warn
                    print(f"  ⚠️ Warning: could not cast column {c} to int cleanly. Values may be non-integer.")
        else:
            print(f"  ⚠️ Column {c} not present in dataframe.")

def find_inconsistencies(df: pd.DataFrame, label_col: str):
    """
    Returns:
      inconsistent_keys_df: DataFrame with columns subject_id, hadm_id, n_unique, unique_values_list
      full_rows_df: rows from original df that belong to any inconsistent group
    """
    if label_col not in df.columns:
        print(f"  - label column {label_col} not found in parquet. Skipping.")
        return pd.DataFrame(), pd.DataFrame()

    # normalize types to avoid grouping surprises
    df['_label_for_check'] = df[label_col].where(pd.notna(df[label_col]), pd.NA)

    # compute unique values per group
    agg = df.groupby(['subject_id','hadm_id'], dropna=False)['_label_for_check'] \
            .agg(lambda s: pd.Series(s.dropna().unique().tolist())) \
            .reset_index()

    # The above yields lists; convert to a list-of-unique-values string and count
    def unique_list(x):
        try:
            lst = list(x)
            # flatten nested lists (happens because of the agg lambda)
            flat = []
            for el in lst:
                if isinstance(el, list):
                    flat.extend(el)
                else:
                    flat.append(el)
            # remove NA-like
            flat_clean = [int(v) if pd.notna(v) else v for v in pd.unique(pd.Series(flat))]
            return flat_clean
        except Exception:
            return []

    agg['unique_values'] = agg['_label_for_check'].apply(unique_list)
    agg['n_unique'] = agg['unique_values'].apply(lambda l: len([v for v in l if pd.notna(v)]) if isinstance(l, list) else 0)
    inconsistent = agg[agg['n_unique'] > 1].copy()
    if inconsistent.empty:
        return inconsistent[['subject_id','hadm_id','n_unique','unique_values']], pd.DataFrame()

    # Get full rows for these groups
    keys = set((int(r['subject_id']), int(r['hadm_id'])) for _, r in inconsistent[['subject_id','hadm_id']].iterrows())
    # filter original df for these keys
    mask = df.apply(lambda r: (int(r['subject_id']), int(r['hadm_id'])) in keys, axis=1)
    full_rows = df[mask].drop(columns=['_label_for_check']) if '_label_for_check' in df.columns else df[mask]
    # order and return
    return inconsistent[['subject_id','hadm_id','n_unique','unique_values']], full_rows

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading parquet: {args.parquet}")
    df = load_parquet(args.parquet)
    print(f"Loaded dataframe with {len(df)} rows and columns: {list(df.columns)}")

    # verify basic id columns
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required id columns: {missing}")

    # cast id columns to int if possible
    ensure_int_columns(df, ['subject_id','hadm_id'])

    # label columns to check
    label_cols = []
    if 'y_mort_1yr' in df.columns:
        label_cols.append('y_mort_1yr')
    if 'y_los_7' in df.columns:
        label_cols.append('y_los_7')

    if not label_cols:
        raise ValueError("Parquet contains neither 'y_mort_1yr' nor 'y_los_7'. Nothing to check.")

    results = {}
    all_inconsistent_keys = set()

    for label in label_cols:
        print(f"\nChecking label: {label}")
        inconsistent_keys_df, full_rows_df = find_inconsistencies(df.copy(), label)
        n_inconsistent = 0 if inconsistent_keys_df is None else len(inconsistent_keys_df)
        print(f"  -> Groups with conflicting {label}: {n_inconsistent}")

        out_keys_csv = os.path.join(args.out_dir, f"inconsistent_{label}.csv")
        out_rows_csv = os.path.join(args.out_dir, f"rows_in_inconsistent_groups_{label}.csv")
        if n_inconsistent > 0:
            inconsistent_keys_df.to_csv(out_keys_csv, index=False)
            # write full rows for manual inspection
            full_rows_df.to_csv(out_rows_csv, index=False)
            print(f"  Wrote {out_keys_csv} and {out_rows_csv}")
            # collect keys
            for _, r in inconsistent_keys_df.iterrows():
                all_inconsistent_keys.add((int(r['subject_id']), int(r['hadm_id'])))
        else:
            # still write empty CSV for clarity
            inconsistent_keys_df.to_csv(out_keys_csv, index=False)
            print(f"  Wrote empty report {out_keys_csv}")

        results[label] = (inconsistent_keys_df, full_rows_df)

    # union report (any label inconsistent)
    if all_inconsistent_keys:
        print(f"\nTotal distinct (subject_id, hadm_id) with any inconsistency: {len(all_inconsistent_keys)}")
        union_keys_df = pd.DataFrame([{'subject_id':k[0],'hadm_id':k[1]} for k in sorted(all_inconsistent_keys)])
        union_keys_path = os.path.join(args.out_dir, 'inconsistent_any_label.csv')
        union_keys_df.to_csv(union_keys_path, index=False)
        # write raw rows for union
        mask = df.apply(lambda r: (int(r['subject_id']), int(r['hadm_id'])) in all_inconsistent_keys, axis=1)
        df[mask].to_csv(os.path.join(args.out_dir, 'rows_in_inconsistent_groups_any_label.csv'), index=False)
        print(f"Wrote union reports to {args.out_dir}")
    else:
        print("\nNo inconsistent groups found across checked labels.")
        # produce empty union file
        pd.DataFrame(columns=['subject_id','hadm_id']).to_csv(os.path.join(args.out_dir, 'inconsistent_any_label.csv'), index=False)

    # Print a few examples to the console for convenience
    for label in label_cols:
        inconsistent_keys_df, full_rows_df = results.get(label, (pd.DataFrame(), pd.DataFrame()))
        if inconsistent_keys_df is None or inconsistent_keys_df.empty:
            print(f"\nNo {label} inconsistencies detected.")
        else:
            print(f"\nExamples of inconsistent groups for {label} (up to 5):")
            print(inconsistent_keys_df.head(5).to_string(index=False))
            print("\nExample raw rows for the first inconsistent group:")
            first = inconsistent_keys_df.iloc[0]
            ex_mask = (df['subject_id']==int(first['subject_id'])) & (df['hadm_id']==int(first['hadm_id']))
            print(df[ex_mask].head(10).to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check downstream parquet for label inconsistencies per (subject_id, hadm_id).")
    parser.add_argument('--parquet', required=True, help="Path to downstream_idx.parquet")
    parser.add_argument('--out_dir', required=True, help="Directory to write inconsistency reports")
    args = parser.parse_args()
    main(args)
