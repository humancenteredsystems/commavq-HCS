"""
Debug rANS encoding for the first segment in the rans archive.

This script:
- Reads the manifest from compression_challenge_submission_rans.zip
- Loads the first segment's payload (should be .rans)
- Uses our decoder to decode symbols
- Loads ground-truth tokens from HF dataset for that segment id
- Compares lengths, prints few mismatched positions and some diagnostics
"""
from pathlib import Path
import zipfile
import json
import numpy as np
import sys

ARCHIVE = "compression_challenge_submission_rans.zip"

try:
    from compression.coder.range_coder import decode_with_cdf
except Exception as e:
    print("Failed importing rans decoder:", e)
    raise

try:
    from datasets import load_dataset
except Exception as e:
    print("Failed importing datasets:", e)
    load_dataset = None

def main():
    zpath = Path(ARCHIVE)
    if not zpath.exists():
        print("Archive not found:", zpath)
        return 2

    with zipfile.ZipFile(str(zpath), "r") as z:
        manifest = json.loads(z.read("manifest.json").decode())
        segments = manifest.get("dataset", {}).get("segments", [])
        if not segments:
            print("No segments in manifest")
            return 3
        seg = segments[0]
        seg_id = seg.get("id")
        rel = seg.get("output_relpath")
        print("First segment:", seg_id, "relpath:", rel)
        payload = z.read(rel)
    # decode
    try:
        symbols = decode_with_cdf(payload)
    except Exception as e:
        print("Decoder failed:", e)
        raise

    print("Decoded symbols length:", len(symbols))
    # reshape
    shape = tuple(seg.get("shape", [1200,8,16]))
    expected_len = shape[0]*shape[1]*shape[2]
    print("Expected token count:", expected_len)
    if len(symbols) != expected_len:
        print("Length mismatch: decoded vs expected")
    arr = np.array(symbols, dtype=np.int32).reshape(shape)
    # load ground truth if possible
    if load_dataset is None:
        print("datasets not available; skipping GT compare")
        return 0
    data_files = manifest.get("cli_args", {}).get("data_files", ['data-0000.tar.gz','data-0001.tar.gz'])
    ds = load_dataset("commaai/commavq", data_files={'train': data_files})
    gt = None
    for i in range(len(ds['train'])):
        row = ds['train'][i]
        try:
            name = row['json']['file_name']
        except Exception:
            name = row.get('__key__')
        if Path(name).name == seg_id:
            gt = np.array(row['token.npy'])
            break
    if gt is None:
        print("GT not found in HF mapping for this seg")
        return 0
    print("GT shape:", gt.shape)
    eq = np.array_equal(gt, arr)
    print("Exact equality:", eq)
    if not eq:
        diff = (gt != arr)
        idxs = diff.ravel().nonzero()[0]
        print("Total diffs:", len(idxs))
        for k in range(min(10, len(idxs))):
            idx = idxs[k]
            t = idx // (shape[1]*shape[2])
            rem = idx % (shape[1]*shape[2])
            h = rem // shape[2]
            w = rem % shape[2]
            print(f"diff #{k}: flat={idx} (t={t},h={h},w={w}) GT={int(gt.ravel()[idx])} REC={int(arr.ravel()[idx])}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
