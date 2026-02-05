#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def iter_records(path):
    """
    Yield each record from either:
    - a .jsonl file (one JSON object per line), or
    - a JSON file with a top-level list.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        # Peek at first non-whitespace char to guess format
        first_non_ws = None
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                break

        f.seek(0)

        if first_non_ws == "[":
            # JSON array
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse JSON array: {e}")
            if not isinstance(data, list):
                raise RuntimeError("Top-level JSON is not a list.")
            for idx, obj in enumerate(data, start=1):
                if not isinstance(obj, dict):
                    print(f"Skipping element {idx}: not a JSON object")
                    continue
                yield idx, obj
        else:
            # JSONL: one object per line
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping line {i}: JSON decode error: {e}")
                    continue
                if not isinstance(obj, dict):
                    print(f"Skipping line {i}: not a JSON object")
                    continue
                yield i, obj


def load_mortality_probs(path):
    probs = []
    labels = []  # optional: ground truth labels if present

    for idx, obj in iter_records(path):
        try:
            pred = obj["predictions"]
            p = pred["mortality_probability"]
        except KeyError:
            print(f"Skipping record {idx}: missing predictions.mortality_probability")
            continue

        if p is None:
            continue

        try:
            p_float = float(p)
        except (TypeError, ValueError):
            print(f"Skipping record {idx}: mortality_probability not a number ({p})")
            continue

        probs.append(p_float)

        # Try to grab ground truth if available
        try:
            gt = obj["ground_truth"]["in_hospital_mortality_48hr"]
            labels.append(int(gt))
        except Exception:
            labels.append(None)

    return probs, labels


def plot_distribution(probs, out_path=None):
    if not probs:
        print("No probabilities to plot.")
        return

    fig, ax = plt.subplots()
    ax.hist(probs, bins=20, range=(0.0, 1.0), alpha=0.8, edgecolor="black")

    ax.set_xlabel("mortality_probability")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of mortality_probability (n={len(probs)})")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot distribution of mortality_probability from JSON/JSONL."
    )
    parser.add_argument("input_path", help="Path to .jsonl or .json file")
    parser.add_argument(
        "--out",
        "-o",
        help="Optional output image path (e.g. plot.png). If omitted, show interactively.",
    )
    args = parser.parse_args()

    probs, labels = load_mortality_probs(args.input_path)
    print(f"Loaded {len(probs)} mortality_probability values.")

    plot_distribution(probs, args.out)


if __name__ == "__main__":
    main()
