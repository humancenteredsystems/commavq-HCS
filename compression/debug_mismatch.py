"""
Debug tool: compare a reconstructed segment to HF ground-truth and print mismatch summary.

Usage:
  python -m compression.debug_mismatch <segment_id> [--archive archive.zip] [--recon_dir reconstructed_rans]

Example:
  python -m compression.debug_mismatch 3b41c0fa8959aea6c118e5714f412a2e_13 --archive compression_challenge_submission_rans.zip --recon_dir reconstructed_rans
"""
import argparse
import json
import zipfile
from pathlib import Path
import numpy as np

try:
    from datasets import load_dataset
except Exception as e:
    print("datasets not available:", e)
    raise

def load_manifest_cli(archive_path: Path):
    with zipfile.ZipFile(str(archive_path), "r") as z:
        m = json.loads(z.read("manifest.json").decode())
    return m.get("cli_args", {}), m.get("dataset", {}).get("segments", [])

def find_gt_by_id(seg_id: str, data_files, split="train"):
    ds = load_dataset("commaai/commavq", data_files={split: data_files})
    for i in range(len(ds[split])):
        row = ds[split][i]
        try:
            name = row["json"]["file_name"]
        except Exception:
            name = row.get("__key__", f"segment-{i:06d}.npy")
        if Path(name).name == seg_id:
            return np.array(row["token.npy"])
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("segment_id", type=str)
    p.add_argument("--archive", type=str, default="compression_challenge_submission_rans.zip")
    p.add_argument("--recon_dir", type=str, default="reconstructed_rans")
    p.add_argument("--split", type=str, default="train")
    args = p.parse_args()

    seg_id = args.segment_id
    archive = Path(args.archive)
    recon_dir = Path(args.recon_dir)

    if not archive.exists():
        print("Archive not found:", archive)
        raise SystemExit(2)

    cli_args, segments = load_manifest_cli(archive)
    data_files = cli_args.get("data_files") or ['data-0000.tar.gz','data-0001.tar.gz']
    print("Using data_files from manifest / fallback:", data_files)

    print("Locating ground truth in HF dataset (may take a moment)...")
    gt = find_gt_by_id(seg_id, data_files, split=args.split)
    if gt is None:
        print("Ground truth for segment not found in HF mapping:", seg_id)
    else:
        print("Ground truth found. shape:", gt.shape)

    # load reconstructed file
    recon_path = recon_dir / (seg_id + ".npy")
    if not recon_path.exists():
        print("Reconstructed file not found at", recon_path)
        raise SystemExit(3)
    rec = np.load(str(recon_path), allow_pickle=False)
    print("Reconstructed shape:", rec.shape)

    if gt is None:
        print("No GT to compare. Exiting.")
        return

    if gt.shape != rec.shape:
        print("Shape mismatch: GT", gt.shape, "REC", rec.shape)

    equal = np.array_equal(gt, rec)
    print("Exact equality:", equal)
    if not equal:
        diff = (gt != rec)
        diff_flat = diff.ravel()
        idxs = np.nonzero(diff_flat)[0]
        print("Total differing positions:", len(idxs))
        # show first few diffs
        N = min(10, len(idxs))
        for k in range(N):
            idx = idxs[k]
            gt_val = int(gt.ravel()[idx])
            rec_val = int(rec.ravel()[idx])
            # convert flat idx to (t,h,w)
            # tokens shape is (1200,8,16)
            t = idx // (8*16)
            rem = idx % (8*16)
            h = rem // 16
            w = rem % 16
            print(f"diff #{k}: flat_idx={idx} (t={t},h={h},w={w}) GT={gt_val} REC={rec_val}")
        # print small surrounding patch example around first diff
        first = idxs[0]
        t = first // (8*16)
        print(f"Example frame index of first diff: {t}")
        # print counts distribution for gt vs rec on that frame
        gt_frame = gt[t].ravel()
        rec_frame = rec[t].ravel()
        import collections
        print("GT frame value counts (top 10):", collections.Counter(gt_frame).most_common(10))
        print("REC frame value counts (top 10):", collections.Counter(rec_frame).most_common(10))

if __name__ == "__main__":
    main()
