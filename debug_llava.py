import os

# The specific problematic paths for Llava
paths_to_check = [
    "/scratch/baj321/MedAgent/results/Llava/cot-sc-ps",
    "/scratch/baj321/MedAgent/results/Llava/fewshot-ps"
]

print("--- DEBUGGING LLAVA FOLDERS ---")

for p in paths_to_check:
    print(f"\nChecking: {p}")
    if os.path.exists(p):
        files = os.listdir(p)
        if not files:
            print("  > Folder is EMPTY.")
        else:
            for f in files:
                print(f"  > Found: {f}")
    else:
        print("  > Folder DOES NOT EXIST.")