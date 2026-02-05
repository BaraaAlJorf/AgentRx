#!/usr/bin/env python3

from pathlib import Path
import re
import sys

if len(sys.argv) != 2:
    print("Usage: ./replace_los.py /path/to/mortality_1yr")
    sys.exit(1)

root = Path(sys.argv[1]).resolve()
if not root.is_dir():
    print(f"ERROR: {root} is not a directory")
    sys.exit(1)

patterns = [
    # --task los / --task "los" / --task=los  → mortality_1yr
    (re.compile(r'(--task\s*(?:=|\s)\s*)([\'"]?)(los)([\'"]?)', re.IGNORECASE),
     r'\1"mortality_1yr"'),

    # dataset split directory
    (re.compile(r'los_multimodal_dataset_splits', re.IGNORECASE),
     'mortality_1yr_multimodal_dataset_splits'),

    # results/los → results/mortality_1yr
    (re.compile(r'(?i)results/los\b'),
     'results/mortality_1yr'),

    # /los/ or /LOS/ → /mortality_1yr/
    (re.compile(r'/(?i:los)(?=/)'),
     '/mortality_1yr'),

    # ./los/ → ./mortality_1yr/
    (re.compile(r'(?<=\./)(?i:los)(?=/)'),
     'mortality_1yr'),
]

changed = 0

for sh_file in root.rglob("*.sh"):
    try:
        text = sh_file.read_text()
    except Exception as e:
        print(f"Skipping {sh_file}: {e}")
        continue

    new_text = text
    for pat, repl in patterns:
        new_text = pat.sub(repl, new_text)

    if new_text != text:
        sh_file.write_text(new_text)
        changed += 1
        print(f"Updated: {sh_file}")

print(f"\nDone. Modified {changed} .sh file(s).")
