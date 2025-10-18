"""
Inspect HF dataset file_name mapping and compare against manifest segments.

Usage:
  python compression/hf_inspect.py --archive ./compression_challenge_submission.zip --limit 200

This script:
- Reads manifest.json from the archive.
- Loads the HF dataset using the data_files listed in manifest.cli_args.
- Prints the first N file names (limit) and their indices.
- Builds a mapping of file_name -> dataset index for the split and reports which manifest segment ids are present/missing.
"""
import argparse
import zipfile
import json
from pathlib import Path
import numpy as np

try:
    from datasets import load_dataset
except Exception as e:
    print("datasets not available:", e)
    raise

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()

    z = zipfile.ZipFile(args.archive, "r")
    manifest = json.loads(z.read("manifest.json").decode())
    cli_args = manifest.get("cli_args", {})
    data_files = cli_args.get("data_files", [])
    print("Manifest data_files:", data_files)
    if not data_files:
        print("No data_files in manifest; exiting.")
        return

    print("Loading HF dataset (this may take a moment)...")
    ds = load_dataset("commaai/commavq", data_files={args.split: data_files})
    split = ds[args.split]
    total = len(split)
    print(f"Loaded split '{args.split}' with {total} rows")

    # print first N file names
    print(f"First {args.limit} rows (index: file_name):")
    for i in range(min(args.limit, total)):
        row = split[i]
        name = None
        try:
            name = row["json"]["file_name"]
        except Exception:
            name = row.get("__key__", f"segment-{i:06d}.npy")
        print(f"{i}: {name}")

    # build mapping for entire split from safe name -> index
    mapping = {}
    for i in range(total):
        row = split[i]
        try:
            name = row["json"]["file_name"]
        except Exception:
            name = row.get("__key__", f"segment-{i:06d}.npy")
        mapping[Path(name).name] = i

    manifest_segments = [s.get("id") for s in manifest.get("dataset", {}).get("segments", [])]
    present = [s for s in manifest_segments if Path(s).name in mapping]
    missing = [s for s in manifest_segments if Path(s).name not in mapping]

    print(f"Manifest segments: {len(manifest_segments)}; present in HF mapping: {len(present)}; missing: {len(missing)}")
    if missing:
        print("Missing examples (first 20):", missing[:20])
    else:
        print("All manifest segments were found in HF dataset mapping.")

if __name__ == '__main__':
    main()
