#!/usr/bin/env python3
"""
Evaluate decompressed archive.

Features:
- Default behavior: verify decompressed files against the HF dataset (like previous script).
- --from-manifest: compute metrics (and subset size) from manifest.json inside the archive so subset runs produce correct metrics.
- Produces a JSON report when --metrics-json is specified.

Usage examples:
  # Legacy-style (full HF split)
  python -m compression.evaluate --archive ./compression_challenge_submission.zip --unpacked ./reconstructed

  # Subset-aware (use manifest to compute numerator)
  python -m compression.evaluate --archive ./compression_challenge_submission.zip --unpacked ./reconstructed --from-manifest --metrics-json report.json
"""
from pathlib import Path
import argparse
import json
import zipfile
import os
import multiprocessing
import numpy as np
from typing import List, Dict

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

TOKENS_PER_SEGMENT = 1200 * 128  # frames * tokens_per_frame
BITS_PER_TOKEN = 10


def load_manifest_segments(archive_path: Path) -> List[Dict]:
    """Read manifest.json from archive and return dataset['segments'] list."""
    with zipfile.ZipFile(str(archive_path), "r") as z:
        raw = z.read("manifest.json").decode("utf-8")
        manifest = json.loads(raw)
    return manifest.get("dataset", {}).get("segments", []), manifest.get("cli_args", {})


def build_hf_mapping(data_files: List[str], split_name: str) -> Dict[str, np.ndarray]:
    """
    Load HF dataset and return mapping safe_filename -> token array.
    If datasets isn't available this raises.
    """
    ds = load_dataset("commaai/commavq", data_files={split_name: data_files})
    mapping = {}
    for i in range(len(ds[split_name])):
        row = ds[split_name][i]
        try:
            name = row["json"]["file_name"]
        except Exception:
            name = row.get("__key__", f"segment-{i:06d}.npy")
        safe = Path(name).name
        mapping[safe] = np.array(row["token.npy"])
    return mapping


def compare_segments(manifest_ids: List[str], unpacked_dir: Path, hf_mapping: Dict[str, np.ndarray]):
    """
    Compare each manifest id's reconstructed file under unpacked_dir to HF mapping if present.
    Returns a report dict with counts and example lists.
    """
    missing_files = []
    no_gt = []
    mismatches = []
    compared = 0
    for seg_id in manifest_ids:
        safe = Path(seg_id).name
        candidate = unpacked_dir / safe
        if not candidate.exists():
            candidate_npy = candidate.with_suffix(".npy")
            if candidate_npy.exists():
                candidate = candidate_npy
            else:
                missing_files.append(safe)
                continue
        # attempt to load
        try:
            arr = np.load(str(candidate), allow_pickle=False)
        except Exception as e:
            missing_files.append(safe)
            continue
        # compare if ground-truth exists
        if hf_mapping and safe in hf_mapping:
            compared += 1
            if not np.array_equal(arr, hf_mapping[safe]):
                mismatches.append(safe)
        else:
            no_gt.append(safe)
    return {
        "missing_files": missing_files,
        "no_ground_truth": no_gt,
        "mismatches": mismatches,
        "compared": compared,
    }


def main(argv: List[str]):
    p = argparse.ArgumentParser(description="Evaluate decompressed archive and compute compression metrics.")
    p.add_argument("--archive", type=str, default="./compression_challenge_submission.zip", help="Packed archive path")
    p.add_argument("--unpacked", type=str, default="./compression_challenge_submission_decompressed/", help="Directory with decompressed .npy files")
    p.add_argument("--from-manifest", action="store_true", help="Compute numerator (segments) from manifest.json inside archive (subset-aware).")
    p.add_argument("--split_name", type=str, default="train", help="HF split name when loading dataset")
    p.add_argument("--metrics-json", type=str, default=None, help="Optional path to write JSON metrics report")
    args = p.parse_args(argv)

    archive_path = Path(args.archive)
    unpacked_dir = Path(args.unpacked)

    if not archive_path.exists():
        print(f"Archive not found: {archive_path}")
        raise SystemExit(2)

    # Default numerator: full HF dataset size (old behaviour)
    total_segments = None
    hf_cli_args = {}

    if args.from_manifest:
        segments, cli_args = load_manifest_segments(archive_path)
        total_segments = len(segments)
        manifest_ids = [s.get("id") for s in segments]
        hf_cli_args = cli_args or {}
        print(f"Using manifest: {archive_path} -> {total_segments} segments")
    else:
        # fallback: try to use HF dataset to determine total rows
        if load_dataset is None:
            print("datasets library not available; cannot determine full dataset size. Use --from-manifest for subset-aware metrics.")
            raise SystemExit(3)
        data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
        ds = load_dataset('commaai/commavq', data_files=data_files)
        total_segments = sum(ds.num_rows.values())
        manifest_ids = None
        hf_cli_args = {"data_files": data_files['train']}

    # compute archive size
    archive_size = archive_path.stat().st_size

    # If verification is desired, attempt to compare reconstructed files to HF ground truth where possible.
    hf_mapping = {}
    hf_available = False
    if load_dataset is not None:
        # prefer manifest.cli_args.data_files if available
        data_files = hf_cli_args.get("data_files") if hf_cli_args.get("data_files") else ['data-0000.tar.gz', 'data-0001.tar.gz']
        try:
            print("Loading HF dataset mapping (may take time)...")
            hf_mapping = build_hf_mapping(data_files, args.split_name)
            hf_available = True
            print(f"HF mapping loaded: {len(hf_mapping)} entries")
        except Exception as e:
            print(f"Warning: failed to load HF mapping for verification: {e}")
            hf_mapping = {}
            hf_available = False

    # If manifest ids are provided, compare those; otherwise compare all hf_mapping keys found in unpacked dir
    if manifest_ids is not None:
        report_cmp = compare_segments(manifest_ids, unpacked_dir, hf_mapping)
    else:
        # build list of candidates from unpacked folder
        files = [p.name for p in unpacked_dir.iterdir() if p.suffix == ".npy"]
        report_cmp = compare_segments(files, unpacked_dir, hf_mapping)

    # Compute tokens and rate using total_segments (manifest-based if requested)
    tokens_total = total_segments * TOKENS_PER_SEGMENT
    bytes_total = archive_size
    rate = (tokens_total * (BITS_PER_TOKEN / 8.0)) / bytes_total if bytes_total > 0 else 0.0
    bpt = (bytes_total * 8.0) / tokens_total if tokens_total > 0 else float("inf")

    metrics = {
        "archive": str(archive_path),
        "archive_bytes": bytes_total,
        "segments_count": total_segments,
        "tokens_total": tokens_total,
        "compression_rate": rate,
        "bits_per_token": bpt,
        "verification": report_cmp,
    }

    print("Evaluation summary:")
    print(f"  Archive: {archive_path}")
    print(f"  Archive size (bytes): {bytes_total}")
    print(f"  Segments counted: {total_segments}")
    print(f"  Tokens total: {tokens_total:,}")
    print(f"  Compression rate (tokens*10/8 / bytes): {rate:.3f}")
    print(f"  Bits per token (archive_bytes*8 / tokens): {bpt:.6f}")
    # verification summary
    print("Verification summary:")
    print(f"  compared (with GT): {report_cmp['compared']}")
    print(f"  mismatches: {len(report_cmp.get('mismatches', []))}")
    print(f"  missing files: {len(report_cmp.get('missing_files', []))}")
    print(f"  no ground truth (skipped): {len(report_cmp.get('no_ground_truth', []))}")

    if args.metrics_json:
        try:
            with open(args.metrics_json, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Wrote metrics JSON to {args.metrics_json}")
        except Exception as e:
            print(f"Warning: failed to write metrics json: {e}")

    # exit code: non-zero if mismatches found
    if len(report_cmp.get("mismatches", [])) > 0:
        raise SystemExit(6)
    else:
        raise SystemExit(0)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
