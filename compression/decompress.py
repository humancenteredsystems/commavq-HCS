#!/usr/bin/env python3
"""
Stage 0 decompressor (submission artifact).

Reads an archive created by compression.compress (stage 0) and writes out reconstructed token .npy files.
By default runs in CPU-only deterministic mode. Optionally verifies against the HF dataset when --verify is passed.

Usage:
  python -m compression.decompress --input ./compression_challenge_submission.zip --output_dir ./reconstructed --verify
"""
from pathlib import Path
import argparse
import sys
import io
import os
import numpy as np

from compression.io import ArchiveReader

try:
    import lzma
except Exception:
    lzma = None

# optional import for verification
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def _np_load_bytes(b: bytes) -> np.ndarray:
    """
    Load a numpy array from .npy bytes (no pickle).
    """
    buf = io.BytesIO(b)
    return np.load(buf, allow_pickle=False)


def _maybe_decompress_payload(payload: bytes, relpath: str) -> bytes:
    """
    If payload is lzma-compressed (by extension .lzma), decompress it.
    Otherwise return payload unchanged.
    """
    if relpath.endswith(".lzma") or relpath.endswith(".xz"):
        if lzma is None:
            raise RuntimeError("lzma module not available to decompress payload")
        return lzma.decompress(payload)
    return payload


def main(argv):
    p = argparse.ArgumentParser(description="Stage0 decompressor: extract and reconstruct token .npy files from archive")
    p.add_argument("--input", required=True, help="Input archive path (zip)")
    p.add_argument("--output_dir", required=True, help="Directory to write reconstructed .npy files")
    p.add_argument("--verify", action="store_true", help="If set, verify reconstructed files against HF dataset (requires datasets lib)")
    p.add_argument("--split_name", default="train", help="Dataset split name used for verification")
    p.add_argument("--allow-missing-gt", action="store_true", help="If set, do not fail when manifest segments are missing ground truth in HF; only verify available ones")
    p.add_argument("--verify-json", type=str, default=None, help="Optional path to write JSON verification report")
    p.add_argument("--max-report", type=int, default=10, help="Maximum number of example IDs to include in printed reports")
    args = p.parse_args(argv)

    reader = ArchiveReader(args.input)
    manifest = reader.manifest()
    # validate scan_order if present
    scan_order = manifest.get("scan_order")
    if scan_order:
        try:
            from compression.scan_order import permutation_sha256_bytes, get_spatial_permutation
            spatial = scan_order.get("spatial", {})
            name = spatial.get("name")
            perm = spatial.get("permutation", [])
            width = spatial.get("width")
            height = spatial.get("height")
            logger = __import__('logging').getLogger('compression.decompress')
            logger.info(f"Found scan_order in manifest: {scan_order.get('traversal')} / spatial={name} {width}x{height}")
            # basic validation
            if width != 16 or height != 8:
                raise RuntimeError(f"Unsupported spatial dims in scan_order: {width}x{height}")
            if len(perm) != 128:
                raise RuntimeError(f"Scan order permutation length != 128: {len(perm)}")
            # verify sha256
            sha = spatial.get('permutation_sha256')
            calc = permutation_sha256_bytes(perm)
            if sha != calc:
                raise RuntimeError(f"Scan order permutation sha mismatch: manifest={sha} calc={calc}")
            # verify permutation is a valid permutation of 0..127
            if sorted(perm) != list(range(128)):
                raise RuntimeError("Scan order permutation is not a full permutation of 0..127")
            logger.info(f"Scan order validated: {name} (sha256={sha})")
        except Exception as e:
            print(f"Warning: scan_order validation failed: {e}")
            # continue anyway for Stage 0
            pass

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = manifest.get("dataset", {}).get("segments", [])
    if not segments:
        # Fallback: list streams and extract everything
        print("No segments listed in manifest.dataset.segments; extracting all streams into output_dir")
        reader.extract_all(str(out_dir))
        reader.close()
        print("Extraction complete.")
        return

    # iterate segments and reconstruct
    failed = []
    for seg in segments:
        seg_id = seg.get("id")
        relpath = seg.get("output_relpath")
        if relpath is None:
            # try to infer path
            relpath = seg.get("path") or f"data/{seg_id}"
        try:
            payload = reader.read_stream_bytes(relpath)
        except KeyError:
            print(f"Stream not found in archive: {relpath}")
            failed.append(seg_id)
            continue
        # maybe decompress per-segment payload
        payload = _maybe_decompress_payload(payload, relpath)
        try:
            # Support custom coder payloads (e.g., .rans) as well as raw .npy/.lzma
            if relpath.endswith(".rans"):
                # decode using our range coder and reshape to original segment shape
                try:
                    from compression.coder.range_coder import decode_with_cdf
                except Exception:
                    raise RuntimeError("rans decoder not available in this installation")
                symbols = decode_with_cdf(payload)
                # obtain shape from manifest entry if present
                shape = seg.get("shape")
                if shape is None:
                    # default fallback
                    shape = (1200, 8, 16)
                # ensure shape is a tuple
                shape = tuple(shape)
                arr = np.array(symbols, dtype=np.int16).reshape(shape)
            else:
                arr = _np_load_bytes(payload)
        except Exception as e:
            print(f"Failed to load numpy array for segment {seg_id} from {relpath}: {e}")
            failed.append(seg_id)
            continue
        # write out canonical .npy file under output_dir with seg_id as filename
        # ensure seg_id is file-system safe
        safe_name = Path(seg_id).name
        out_path = out_dir / safe_name
        # if seg_id already has an extension like .npy, preserve it; otherwise add .npy
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".npy")
        np.save(str(out_path), arr, allow_pickle=False)
    reader.close()

    if failed:
        print(f"Completed with {len(failed)} failures. Failed segments: {failed[:10]}")
    else:
        print(f"Reconstructed {len(segments)} segments to {out_dir}")

    # verification step (optional)
    if args.verify:
        if load_dataset is None:
            print("datasets library not available; cannot verify.")
            print("Set --allow-missing-gt to skip strict HF checks when running offline or on partial archives.")
            return
        print("Running verification against HF dataset (this requires network/access and may be slow)...")
        data_files = {args.split_name: manifest.get("cli_args", {}).get("data_files", []) or []}
        if not any(data_files[args.split_name]):
            print("No data_files listed in manifest.cli_args; cannot load HF dataset for verification.")
            print("Set --allow-missing-gt to skip strict HF checks when running offline or on partial archives.")
            return
        # Build the list of segments present in the manifest and verify only those.
        segments_to_check = [s.get("id") for s in manifest.get("dataset", {}).get("segments", [])]
        if not segments_to_check:
            print("No segments listed in manifest to verify against. Skipping verification.")
            return

        # Load HF dataset and build mapping of available ground-truth tokens for manifest segments.
        hf_tokens = {}
        hf_missing = []
        try:
            ds = load_dataset("commaai/commavq", data_files=data_files)
            needed = set(Path(s).name for s in segments_to_check)
            for i in range(len(ds[args.split_name])):
                row = ds[args.split_name][i]
                try:
                    name = row["json"]["file_name"]
                except Exception:
                    name = row.get("__key__", f"segment-{i:06d}.npy")
                safe_name = Path(name).name
                if safe_name in needed:
                    hf_tokens[safe_name] = np.array(row["token.npy"])
                    # early exit if we've gathered all
                    if len(hf_tokens) >= len(needed):
                        break
            # compute which manifest segments lacked GT in HF mapping
            hf_missing = [s for s in segments_to_check if Path(s).name not in hf_tokens]
        except Exception as e:
            print(f"Warning: failed to load HF dataset for verification: {e}")
            hf_missing = list(segments_to_check)  # mark all as missing GT

        # If there are missing ground-truth segments and the user did not allow missing GT, fail.
        if hf_missing and not args.allow_missing_gt:
            n = len(hf_missing)
            print(f"Verification FAILED: {n} segment(s) from manifest have no ground-truth in the HF mapping.")
            print(f"Examples (first {args.max_report}): {hf_missing[:args.max_report]}")
            print("If you are intentionally verifying a partial archive or are offline, re-run with --allow-missing-gt to verify only available segments.")
            # optionally write JSON report
            if args.verify_json:
                try:
                    import json as _json
                    report = {
                        "total_manifest_segments": len(segments_to_check),
                        "ground_truth_available": len(hf_tokens),
                        "ground_truth_missing": len(hf_missing),
                        "missing_examples": hf_missing[:args.max_report],
                    }
                    with open(args.verify_json, "w") as _f:
                        _json.dump(report, _f, indent=2)
                except Exception:
                    pass
            import sys as _sys
            _sys.exit(6)

        # Now verify arrays for the segments that do have ground-truth.
        mismatches = []
        compared = 0
        for seg_id in segments_to_check:
            safe_name = Path(seg_id).name
            rec_path = out_dir / safe_name
            if not rec_path.exists():
                rec_path_npy = rec_path.with_suffix(".npy")
                if rec_path_npy.exists():
                    rec_path = rec_path_npy
                else:
                    mismatches.append((seg_id, "missing_file"))
                    continue
            if safe_name in hf_tokens:
                compared += 1
                got = np.load(str(rec_path), allow_pickle=False)
                expected = hf_tokens[safe_name]
                if not np.array_equal(got, expected):
                    mismatches.append((seg_id, "mismatch"))
            else:
                # no GT for this segment; skip (user allowed this or --allow-missing-gt used)
                continue

        # Build and optionally write a JSON report
        report = {
            "total_manifest_segments": len(segments_to_check),
            "ground_truth_available": len(hf_tokens),
            "ground_truth_missing": len(hf_missing),
            "compared": compared,
            "mismatches": len(mismatches),
            "mismatch_examples": mismatches[:args.max_report],
        }
        if args.verify_json:
            try:
                import json as _json
                with open(args.verify_json, "w") as _f:
                    _json.dump(report, _f, indent=2)
            except Exception:
                pass

        # Final decision based on mismatches
        if report["mismatches"] > 0:
            print(f"Verification FAILED: {report['mismatches']} mismatched segments out of {report['compared']} compared.")
            print(f"Examples (first {args.max_report}): {report['mismatch_examples']}")
            import sys as _sys
            _sys.exit(6)
        else:
            print("Verification successful for all segments listed in the manifest (where ground truth was available).")
            # print a short summary
            print(f"Manifest segments: {report['total_manifest_segments']}; compared: {report['compared']}; missing GT: {report['ground_truth_missing']}")


if __name__ == "__main__":
    main(sys.argv[1:])
