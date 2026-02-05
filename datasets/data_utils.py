import json
import os
import csv
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# --- Shared EHR Formatting Function ---
# def format_ehr_as_text(path: str) -> str:
#     """Generates 48 1-hour bins for all variables."""
#     if not os.path.exists(path):
#         return "EHR timeseries data not available."

#     N_BINS = 48
#     TIMESTEP = 1.0
#     MAX_HOURS = 48.0
    
#     feature_names = []
#     binned_values = [] 

#     try:
#         with open(path, 'r', encoding='utf-8-sig') as f:
#             reader = csv.reader(f, delimiter=',')
#             try:
#                 header = next(reader)
#             except StopIteration:
#                 return "EHR file is empty."
                
#             if not header or header[0].lower() != 'hours':
#                  feature_names = header
#             else:
#                  feature_names = header[1:]

#             if not feature_names:
#                 return "EHR file has no feature columns after 'Hours'."

#             N_CHANNELS = len(feature_names)
#             binned_values = [[[] for _ in range(N_BINS)] for _ in range(N_CHANNELS)]

#             for row in reader:
#                 if not row: continue
#                 try:
#                     t = float(row[0])
#                     if t >= MAX_HOURS or t < 0: continue
#                     bin_id = int(t / TIMESTEP)
#                     for i in range(N_CHANNELS):
#                         if i+1 < len(row) and row[i+1] != "":
#                             binned_values[i][bin_id].append(row[i+1])
#                 except ValueError:
#                     continue 
#     except Exception as e:
#         return f"Could not process EHR data: {e}"

#     lines = ["Hourly EHR Vitals and Labs (imputed using 'previous' value):"]
#     prev_values = [None] * N_CHANNELS 
    
#     for i, name in enumerate(feature_names):
#         value_strings = [] 
#         for b in range(N_BINS):
#             bin_data = binned_values[i][b]
#             if len(bin_data) > 0:
#                 current_value_str = bin_data[-1]
#                 prev_values[i] = current_value_str
#                 value_strings.append(current_value_str)
#             else:
#                 if prev_values[i] is not None:
#                     value_strings.append(f"[imp]{prev_values[i]}")
#                 else:
#                     value_strings.append("[NaN]")
#         lines.append(f"- {name.strip()}: [{','.join(value_strings)}]")

#     return "\n".join(lines)

def format_ehr_as_text(path: str) -> str:
    """
    Reads EHR data and formats it as a chronological log.
    Format: "T+2.5h: HR=80, SpO2=98"
    """
    if not os.path.exists(path):
        return "EHR Data: Not Available"

    MAX_HOURS = 48.0
    formatted_lines = ["EHR Clinical Time-Series Log (0-48h):"]
    
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return "EHR Data: Empty file"
            
            # Detect feature names (Skip 'Hours' column)
            if not header:
                return "EHR Data: No headers"
            
            # Assuming format: Hours, Var1, Var2, ...
            feature_names = header[1:]
            
            # Read all rows and filter strictly by time
            # We store valid readings to avoid printing empty lines
            for row in reader:
                if not row: continue
                
                try:
                    t = float(row[0])
                except ValueError:
                    continue # Skip bad rows
                
                if t < 0 or t > MAX_HOURS:
                    continue

                # Collect only the features that exist at this timestamp
                current_measurements = []
                for i, val in enumerate(row[1:]):
                    if val and val.strip(): # If value exists (not empty string)
                        # Clean the value (optional: round numbers if needed)
                        feature_name = feature_names[i].strip()
                        current_measurements.append(f"{feature_name}={val.strip()}")
                
                # Only add this timestep if there is actual data
                if current_measurements:
                    # Format: "[T+1.5h] HR=80, RR=18"
                    line = f"[T+{t:.1f}h] {', '.join(current_measurements)}"
                    formatted_lines.append(line)

    except Exception as e:
        return f"Error processing EHR: {e}"

    # If no data was found in 0-48h window
    if len(formatted_lines) == 1:
        return "EHR Data: No recordings found in first 48 hours."

    # OPTIONAL: Truncate if too long (Qwen has 32k context, so usually fine)
    # But just in case of high-frequency noise (e.g. 1-minute intervals)
    if len(formatted_lines) > 500: 
        # Strategy: Keep first 100 (admission) + last 400 (most recent trend)
        formatted_lines = formatted_lines[:100] + ["... [Intermediate Data Skipped] ..."] + formatted_lines[-400:]

    return "\n".join(formatted_lines)

def load_few_shot_data(data_path: str, num_shots: int = 4) -> list:
    """
    Loads balanced few-shot data (num_shots/2 per class) AND loads their images.
    """
    shots = []
    try:
        import random
        from PIL import Image # Ensure PIL is available
        import os

        all_lines = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip(): all_lines.append(json.loads(line))
        
        # 1. Separate by Class
        positives = [p for p in all_lines if p['labels']['in_hospital_mortality_48hr'] == 1]
        negatives = [p for p in all_lines if p['labels']['in_hospital_mortality_48hr'] == 0]

        if not positives or not negatives:
            print("[Error] Few-shot data file does not contain both classes.")
            return []

        # 2. Balanced Sampling (e.g., if num_shots=4, take 2 Pos, 2 Neg)
        n_per_class = max(1, num_shots // 2)
        
        # Safety check if we request more than available
        selected_pos = random.sample(positives, min(len(positives), n_per_class))
        selected_neg = random.sample(negatives, min(len(negatives), n_per_class))
        
        # Combine and Shuffle
        selected_data = selected_pos + selected_neg
        random.shuffle(selected_data)

        # 3. Process Text AND Images
        for patient in selected_data:
            # A. Process EHR
            if 'ehr_timeseries_path' in patient:
                patient['ehr_text'] = format_ehr_as_text(patient['ehr_timeseries_path'])
            
            # B. Load Image (PIL) - Re-enabled!
            # We explicitly load the image into RAM here.
            patient['pil_image'] = None
            if 'cxr_image_path' in patient and patient['cxr_image_path'] != 'CXR not available':
                if os.path.exists(patient['cxr_image_path']):
                    try:
                        img = Image.open(patient['cxr_image_path']).convert("RGB")
                        img.load() # Force load into memory
                        patient['pil_image'] = img
                    except Exception as e:
                        print(f"Warning: Failed to load few-shot image: {e}")

            shots.append(patient)
            
    except Exception as e:
        print(f"[Warning] Could not load few-shot data: {e}")
        return []
        
    return shots

# --- 1. The Worker Dataset (Reads from Disk) ---
class ClinicalDataset(Dataset):
    """
    Reads files from disk. Used by the pre-loader to fetch data.
    """
    def __init__(self, data_path: str):
        self.data = []
        try:
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Data file not found at {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data[idx]
        
        # A. Process EHR Text (CPU Intensive)
        if 'ehr_timeseries_path' in patient:
            patient['ehr_text'] = format_ehr_as_text(patient['ehr_timeseries_path'])
        else:
            patient['ehr_text'] = "EHR timeseries data not available."
            
        # B. Load Image into RAM (IO Intensive)
        # We load it here so it's ready in memory for the MemoryDataset
        if 'cxr_image_path' in patient and patient['cxr_image_path'] != 'CXR not available':
            if os.path.exists(patient['cxr_image_path']):
                try:
                    # Open and convert to RGB. 
                    # CRITICAL: We must load() the image data immediately, 
                    # otherwise PIL lazy loads and keeps the file handle open.
                    img = Image.open(patient['cxr_image_path']).convert("RGB")
                    img.load() 
                    patient['pil_image'] = img
                except Exception as e:
                    print(f"Error loading image {patient['cxr_image_path']}: {e}")
                    patient['pil_image'] = None
            else:
                patient['pil_image'] = None
        else:
            patient['pil_image'] = None
            
        return patient

# --- 2. The In-Memory Dataset (No IO) ---
class MemoryDataset(Dataset):
    """
    Holds pre-loaded data in a Python list. Zero IO during access.
    """
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    return batch

def get_data_loader(data_path: str, batch_size: int, num_workers: int) -> DataLoader:
    """
    1. Creates a temporary Parallel Loader to read everything from disk.
    2. Stores it in RAM (MemoryDataset).
    3. Returns a fast loader for the training loop.
    """
    print(f"\n[Pre-Loading] Initializing parallel workers ({num_workers}) to load data into RAM...")
    
    # 1. Create the worker dataset
    disk_dataset = ClinicalDataset(data_path=data_path)
    
    # 2. Create a temporary loader to fetch batches using multiprocessing
    # Note: We use a larger batch size for loading to speed it up
    temp_loader = DataLoader(
        disk_dataset,
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    all_data = []
    
    # 3. Iterate and collect into RAM
    # tqdm shows the progress bar for the "Start Time" loading
    for batch in tqdm(temp_loader, desc="Loading Dataset to RAM"):
        all_data.extend(batch)
        
    print(f"[Pre-Loading] Successfully loaded {len(all_data)} samples into RAM.")
    
    # 4. Wrap in MemoryDataset
    memory_dataset = MemoryDataset(all_data)
    
    # 5. Return the final loader (num_workers=0 because data is already in RAM)
    return DataLoader(
        memory_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0, # Fast access from RAM, no overhead needed
        collate_fn=custom_collate_fn
    )

def get_train_data_loader(
    data_path: str,
    batch_size: int,
    num_workers: int,
    max_train_samples: int = None,
    seed: int = None
) -> DataLoader:
    """
    1. Loads the full dataset into RAM via ClinicalDataset + a temp DataLoader.
    2. If max_train_samples is set and < dataset size, draws a random subset (seeded).
    3. Returns a MemoryDataset-backed DataLoader (num_workers=0).
    """
    print(f"\n[Pre-Loading] Initializing parallel workers ({num_workers}) to load data into RAM...")
    
    # 1. Create the worker dataset
    disk_dataset = ClinicalDataset(data_path=data_path)
    
    # 2. Create a temporary loader to fetch batches using multiprocessing
    temp_loader = DataLoader(
        disk_dataset,
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    print("Max Train Samples:",max_train_samples)
    
    all_data = []
    
    # 3. Iterate and collect into RAM
    for batch in tqdm(temp_loader, desc="Loading Dataset to RAM"):
        all_data.extend(batch)
        
    total = len(all_data)
    print(f"[Pre-Loading] Successfully loaded {total} samples into RAM.")
    
    # 4. If sampling requested, draw a random subset (seeded) — DO NOT PERSIST TO DISK
    if isinstance(max_train_samples, int) and 0 < max_train_samples < total:
        import random as _rnd
        rnd = _rnd.Random(seed)
        indices = rnd.sample(range(total), k=max_train_samples)
        sampled = [all_data[i] for i in indices]
        print(f"[Sampling] Selected {len(sampled)} / {total} random samples (seed={seed}).")
        memory_dataset = MemoryDataset(sampled)
    else:
        memory_dataset = MemoryDataset(all_data)
        if isinstance(max_train_samples, int) and max_train_samples >= total:
            print(f"[Sampling] Requested max_train_samples={max_train_samples} >= dataset size ({total}). Using full dataset.")
        else:
            print("[Sampling] No max_train_samples provided. Using full dataset.")
    
    # 5. Return the final loader (num_workers=0 because data is already in RAM)
    return DataLoader(
        memory_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

