#!/usr/bin/env python3
"""
Stage 0 compressor (streaming, memory-safe, verbose).

Creates a self-describing single archive containing per-segment token files and a manifest.
This version is responsive: it streams files into the archive, reports progress and heartbeats,
and supports a --limit for quick smoke tests.

Usage:
  python -m compression.compress --data_files data-0000.tar.gz data-0001.tar.gz \\
      --output ./compression_challenge_submission.zip --mode pass-through --limit 100 --verbose
"""
from pathlib import Path
import argparse
import io
import json
import os
import sys
import time
import logging
import numpy as np

from datasets import load_dataset

from compression.io import ArchiveBuilder
from compression.scan_order import SCAN_ORDER_MANIFEST

DEFAULT_OUT = "./compression_challenge_submission.zip"


def _np_save_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def build_manifest(base_manifest: dict, streams_meta: list) -> dict:
    m = dict(base_manifest)
    m.setdefault("version", "0.1.0")
    m.setdefault("modules", {"transform_chain": [{"name": "identity", "params": {}}], "model": {"name": "none"}})
    m.setdefault("dataset", {})
    m["integrity"] = {"streams": streams_meta}
    return m


def setup_logging(verbose: bool, logfile: str = None):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s", handlers=handlers)
    # ensure immediate flush behavior
    for h in logging.getLogger().handlers:
        try:
            h.flush = h.flush
        except Exception:
            pass


def main(argv):
    p = argparse.ArgumentParser(description="Stage0 compressor: package token segments into a single archive")
    p.add_argument("--data_files", nargs="+", required=True, help="HuggingFace dataset shard tars (data-*.tar.gz)")
    p.add_argument("--output", default=DEFAULT_OUT, help="Output archive path (zip)")
    p.add_argument("--mode", choices=["pass-through", "lzma"], default="pass-through", help="How to store payloads")
    p.add_argument("--split_name", default="train", help="Dataset split name")
    p.add_argument("--limit", type=int, default=0, help="If >0, only process the first N segments (for testing)")
    p.add_argument("--progress-interval", type=int, default=100, help="Log heartbeat every N segments")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    p.add_argument("--quiet", action="store_true", help="Suppress INFO logs (overrides --verbose)")
    p.add_argument("--logfile", type=str, default=None, help="Optional file to log to")
    args = p.parse_args(argv)

    if args.quiet and args.verbose:
        print("Cannot use --quiet and --verbose together", file=sys.stderr)
        sys.exit(3)

    verbose = bool(args.verbose) and not args.quiet

    setup_logging(verbose, args.logfile)
    logger = logging.getLogger("compression.compress")
    logger.info("Starting Stage 0 compressor")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"CWD: {Path.cwd()}")
    logger.info(f"Args: {args}")

    out_path = Path(args.output).resolve()
    builder = ArchiveBuilder(str(out_path), verbose=verbose)

    base_manifest = {
        "version": "0.1.0",
        "git_hash": os.environ.get("GIT_HASH", ""),
        "cli_args": {
            "mode": args.mode,
            "data_files": args.data_files,
            "split": args.split_name,
        },
        "modules": {
            "transform_chain": [{"name": "identity", "params": {}}],
            "model": {"name": "none"},
            "coder": {"type": "none"},
        },
        "dataset": {"segments": []},
    }

    # load dataset
    data_files = {args.split_name: args.data_files}
    try:
        logger.info("Loading dataset (this may download shards from HF)...")
        ds = load_dataset("commaai/commavq", data_files=data_files)
        total_rows = len(ds[args.split_name])
        logger.info(f"Dataset loaded. {total_rows} segments available in split '{args.split_name}'.")
    except KeyboardInterrupt:
        logger.error("Interrupted while loading dataset")
        sys.exit(130)
    except Exception as e:
        logger.exception("Failed to load dataset. Check network, HF credentials, and data_files paths.")
        sys.exit(2)

    limit = int(args.limit) if args.limit and args.limit > 0 else total_rows
    limit = min(limit, total_rows)
    logger.info(f"Processing up to {limit} segments (limit={args.limit})")

    streams_meta = []
    start_time = time.time()
    last_heartbeat = start_time
    processed = 0
    bytes_written = 0

    try:
        for i in range(limit):
            try:
                row = ds[args.split_name][i]
            except Exception as e:
                logger.exception(f"Failed to read row {i} from dataset")
                raise

            # resolve name
            try:
                name = row["json"]["file_name"]
            except Exception:
                name = row.get("__key__", f"segment-{i:06d}.npy")
            relpath = f"data/{name}"

            tokens = None
            try:
                tokens = np.array(row["token.npy"])
            except Exception as e:
                logger.exception(f"Failed to convert tokens for segment {name} (index {i})")
                raise

            payload = _np_save_bytes(tokens)
            if args.mode == "lzma":
                import lzma
                payload = lzma.compress(payload)
                relpath += ".lzma"

            # write into the archive (streamed)
            try:
                builder.add_stream(relpath, payload)
            except Exception as e:
                logger.exception(f"Failed to write stream {relpath} into archive")
                raise

            base_manifest["dataset"]["segments"].append({"id": name, "shape": list(tokens.shape), "output_relpath": relpath})
            processed += 1
            bytes_written += len(payload)

            # heartbeat logging
            now = time.time()
            if processed % args.progress_interval == 0 or (now - last_heartbeat) > 30:
                elapsed = now - start_time
                per_sec = processed / elapsed if elapsed > 0 else 0.0
                logger.info(f"Progress: processed={processed}/{limit}, bytes_written={bytes_written}, elapsed={elapsed:.1f}s, {per_sec:.2f} seg/s")
                last_heartbeat = now

    except KeyboardInterrupt:
        logger.error("Interrupted by user (KeyboardInterrupt). Finalizing partial archive.")
        try:
            builder.set_manifest(build_manifest(base_manifest, streams_meta))
            builder.finalize()
        except Exception:
            logger.exception("Failed to finalize partial archive on interrupt.")
        sys.exit(130)
    except Exception:
        logger.exception("Fatal error during packaging. Aborting.")
        try:
            builder.cleanup()
        except Exception:
            pass
        sys.exit(4)

    # finalize manifest and archive
    try:
        manifest = build_manifest(base_manifest, streams_meta)
        manifest["scan_order"] = SCAN_ORDER_MANIFEST
        builder.set_manifest(manifest)
        logger.info("Writing manifest and finalizing archive...")
        builder.finalize()
        elapsed = time.time() - start_time
        archive_size = out_path.stat().st_size if out_path.exists() else 0
        tokens_total = processed * 1200 * 128  # approximate token count per segment
        score_proxy = (tokens_total * 10 / 8) / archive_size if archive_size > 0 else 0.0
        logger.info(f"Completed: processed={processed}, elapsed={elapsed:.1f}s, archive={out_path} ({archive_size} bytes), score_proxy={score_proxy:.3f}")
    except Exception:
        logger.exception("Failed to finalize archive")
        try:
            builder.cleanup()
        except Exception:
            pass
        sys.exit(4)

    builder.cleanup()
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
