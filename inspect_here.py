import json
import os

# Pick one specific file that we know exists from your logs
TEST_FILE = "/scratch/baj321/MedAgent/results/los/final_qwen/meta-ps/los_summary.json"

def inspect(filepath):
    print(f"Inspecting: {filepath}")
    
    if not os.path.exists(filepath):
        print("File not found.")
        return

    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"Root Type: {type(data)}")

    if isinstance(data, list):
        print(f"Root is a LIST with {len(data)} items.")
        if len(data) > 0:
            print(f"First item type: {type(data[0])}")
            print(f"First item keys: {data[0].keys()}")
    
    elif isinstance(data, dict):
        print(f"Root is a DICT with {len(data)} keys.")
        keys = list(data.keys())
        print(f"First 5 keys: {keys[:5]}")
        
        # Check the content of the first key
        first_val = data[keys[0]]
        print(f"Value of first key ('{keys[0]}') is type: {type(first_val)}")
        if isinstance(first_val, dict):
             print(f"Keys inside first item: {first_val.keys()}")

inspect(TEST_FILE)