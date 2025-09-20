#!/usr/bin/env python3
# validate_label_mapping.py
"""Validate that all labels map cleanly to the 4 target categories.

Usage:
  python scripts/validate_label_mapping.py /path/to/merged_dataset.csv

Saves a small CSV report showing any unmapped or unexpected labels.
"""
import sys
import os
# When this script is executed directly from the scripts/ folder, the repository root
# may not be on Python's import path. Insert the repo root so local package imports
# like `from cyberprint...` work without requiring PYTHONPATH or installation.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd
from cyberprint.data.merge_datasets import LABEL_MAP
from cyberprint.data.merge_datasets import LABEL_MAP

def validate(path):
    # Support parquet or csv
    if path.lower().endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if 'label' not in df.columns:
        print("No 'label' column found in dataset.")
        return 1
    vals = pd.Series(df['label'].astype(str).fillna(''))
    unique = vals.unique()
    mapped = set(LABEL_MAP.keys()) | set(LABEL_MAP.values())
    unexpected = [u for u in unique if u not in mapped]
    print(f"Total rows: {len(df)}")
    print(f"Unique labels in file: {len(unique)}")
    if unexpected:
        print("Unexpected labels (not in LABEL_MAP or target categories):")
        for u in unexpected:
            print(" -", u)
        # Save a quick report
        out = os.path.join(os.path.dirname(path), 'label_mapping_issues.csv')
        df[df['label'].astype(str).isin(unexpected)].to_csv(out, index=False)
        print("Saved examples to", out)
        return 2
    else:
        print("All labels map to expected categories.")
        return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_label_mapping.py /path/to/merged_dataset.csv")
        sys.exit(1)
    sys.exit(validate(sys.argv[1]))
