# Compression System Architecture (Systems Engineering Document)

Status: Draft v0.2  
Owners: Compression Team  
Applies to: commavq-HCS repository

1. Purpose and scope
- Purpose: Define a modular, open, and staged architecture for a lossless compressor tailored to commaVQ tokens, with a stable CLI and archive format compatible with the competition’s submission requirement (single archive + a Python decompressor script).
- Scope:
  - Lossless compression of 5,000 minutes of VQ-VAE video tokens (int16 arrays of shape 1200×8×16 per segment; symbols are 10-bit indices 0–1023).
  - Deterministic decompression on CPU that reproduces the original tokens exactly.
  - Progressive, stage-by-stage improvements with measurable gains at each stage.
- Non-goals:
  - Training recipes and data curation beyond what’s needed for adapters/calibration.
  - Distributed training infrastructure (can be added later).

2. Background and constraints
- Dataset format:
  - Each “segment” ≈ 1 minute at 20 FPS: 1200 frames × 8 × 16 tokens per frame, stored as int16.
  - Token alphabet size: 1024 (10-bit).
- Competition constraints:
  - Submit a single archive (e.g., .zip) and a Python decompression script that reconstructs the original tokens exactly.
  - Highest compression ratio “score” wins; higher is better.
- Engineering constraints:
  - Deterministic decode (CPU-only path; no nondeterministic kernels).
  - Numerically stable softmax and CDF quantization.
  - Archive fully self-describing (manifest with versions, parameters, checksums).

3. High-level architecture
Pipeline: Transform → ProbModel → CDF → EntropyCoder → Archive  
- Transform (reversible): Applies strictly invertible pre-processing to reduce entropy (e.g., temporal delta, bit-plane coding). Outputs transformed tokens + side-info needed to invert.
- ProbModel: Provides probability mass function (PMF) for the next symbol given context. Implementations:
  - order-0 histogram (Stage 1)
  - pretrained GPT wrapper (Stage 3)
  - classic finite-context model (Stage 6) and/or mixers (Stage 6+)
- CDF: Converts PMF to a monotone, quantized cumulative distribution suitable for entropy coding (e.g., 16-bit).
- EntropyCoder: rANS or range coder turning symbols + CDF into bytes; deterministic and invertible.
- Archive: Packages bytestreams, manifest.json, and side-info (e.g., adapters) into a single file.

4. Command-line interfaces (stable contracts)
4.1 Decompression (submission artifact)
- Entry point: compression/decompress.py
- CLI:
  - python -m compression.decompress --input archive.zip --output_dir ./reconstructed --verify
- Behavior:
  - Loads manifest.json from archive root.
  - Reconstructs Transform, ProbModel (including base model + optional adapter), CDF params, and EntropyCoder.
  - Decodes deterministically on CPU and restores the original tokens to output_dir (segment structure and filenames as recorded in manifest).
  - Verification semantics (see section 6.2) are strict by default; missing ground-truth for manifest segments causes verification to fail unless explicitly allowed.

4.2 Compression (internal tool)
- Entry point: compression/compress.py
- CLI:
  - python -m compression.compress --data_root ./data_shard --manifest_base config/stageN.yaml --output archive.zip
- Behavior:
  - Reads segments deterministically from data_root (local or HF datasets via DatasetReader plugin).
  - Runs Transform → ProbModel → CDF → EntropyCoder over a fixed scan order.
  - Writes archive with manifest.json, data/stream_*.bin, and any required side-info.

4.3 Evaluation (internal tool)
- Entry point: compression/evaluate.py
- CLI:
  - python -m compression.evaluate --archive archive.zip --temp_dir ./tmp --report ./report.json
- Behavior:
  - Runs decompress.py on the archive.
  - Optionally compares to ground truth when available.
  - Reports total bytes, bits per token (bpt), score, and timing.

5. Archive layout and manifest
5.1 Archive layout
- /manifest.json
- /data/stream_0.bin, stream_1.bin, … (interleaved coder streams)
- /models/adapter.bin (optional, quantized adapter weights)
- /tables/cdf_meta.bin (optional, only if not derivable)
- /checksums/*.sha256 (optional)

5.2 Manifest schema (top-level fields)
- version: semantic version for the archive format (e.g., "1.0.0")
- git_hash: code revision used to produce the archive
- cli_args: full argv used during compression (provenance)
- modules:
  - transform_chain: ordered list of {name, params}
  - model: {name, base_id/hash, calibration: {type, params}, adapter: {present, path, quant, size_bytes}}
  - cdf: {precision_bits, epsilon_floor, tail_policy}
  - coder: {type: "rans"|"range", interleaving: int, precision_bits}
- dataset:
  - segments: array of {id, shape, length, output_relpath}
  - scan_order: description of token traversal order (See section 6.1)
- integrity:
  - streams: list of {path, sha256}
  - side_info: list of {path, sha256}
- decode_requirements:
  - cpu_only: true
  - python_version: e.g., ">=3.10"
  - dependencies: minimal list required for decode (numpy, etc.)

6. Module interfaces (stable API contracts)
- Transform
  - apply(tokens: np.ndarray, params: Dict) → (tokens_t: np.ndarray, sideinfo: Dict)
  - invert(tokens_t: np.ndarray, sideinfo: Dict) → tokens
  - Notes: Must be strictly invertible; record params in manifest.
- ProbModel
  - reset(segment_meta: Dict) → None
  - pmf(context: Context) → np.ndarray[float] (size 1024 for symbol-level or size 2 for bit-plane)
  - update(context: Context, symbol: int) → None (for classic models/adaptation)
  - Implementations: Order0Model, GPTWrapperModel, ClassicPPM, Mixer
- CDF
  - quantize(pmf: np.ndarray, precision_bits: int, epsilon: float, tail_policy: str) → CDFTable (uint16/uint32 monotone)
  - dequant rules: fully specified and deterministic
- EntropyCoder
  - encode(symbol: int, cdf: CDFTable, state: CoderState) → None
  - finalize(state) → bytes (or write to stream)
  - decode(cdf: CDFTable, state: CoderState) → symbol
- Archive
  - builder: write_manifest(manifest), add_stream(bytes, name), add_sideinfo(path), finalize(out_path)
  - reader: read_manifest(), stream_iter(), sideinfo_paths()

6.1 Canonical scan order (project canonical default)
- Canonical traversal:
  - traversal: "frame_major" — iterate frames in time order, and for each frame traverse spatial positions in a fixed spatial permutation.
  - spatial permutation: "hilbert_8x16_v1"
    - Built from a 16×16 Hilbert curve (order=4), filtered to the top half (rows y in 0..7), and expressed as row-major indices (y*16 + x).
    - Provides 128 spatial positions per frame (8×16).
    - The permutation is frozen and included in manifest.scan_order as:
      {
        "traversal": "frame_major",
        "frame_order": "increasing",
        "spatial": {
          "name": "hilbert_8x16_v1",
          "width": 16,
          "height": 8,
          "permutation_kind": "row_major_index",
          "permutation_sha256": "<sha256 of LE-int32 bytes>",
          "permutation": [ ... 128 ints ... ]
        }
      }
- Rationale: Hilbert ordering improves locality (spatial stationarity) and is deterministic; recording the permutation and its sha256 ensures reproducible decoding and experiment provenance.
- Implementation: compression/scan_order.py provides SCAN_ORDER_MANIFEST and utilities for validation.

6.2 Verification semantics (strict-by-default)
- Purpose: Provide deterministic and unambiguous verification behavior for decompressed archives.
- Default (strict) behavior:
  - When --verify is requested, the decompressor will:
    1. Read the list of segments from manifest.dataset.segments.
    2. Attempt to load HF ground-truth using manifest.cli_args.data_files (split = manifest.cli_args.split or --split_name).
    3. Build a mapping of HF file_name -> tokens for the manifest segments.
    4. If any manifest segments are missing from the HF mapping (no ground-truth available), verification fails with a non-zero exit code and lists example missing IDs.
    5. If ground-truth is available for a segment, the script compares arrays exactly; any mismatch fails verification with a non-zero exit code and example mismatches in the output.
- Lenient mode for development:
  - Flag: --allow-missing-gt
    - When set, missing ground-truth segments are treated as warnings and verification proceeds by checking only the segments that do have ground-truth.
- Reporting:
  - Flag: --verify-json <path>
    - Writes a structured JSON report with counts: total_manifest_segments, ground_truth_available, ground_truth_missing, compared, mismatches, and example lists (truncated by --max-report).
  - Flag: --max-report N
    - Controls number of example IDs included in printed or JSON reports.

7. Determinism and numerics
- CPU-only decode path; disable nondeterministic kernels.
- Fixed random seeds (encode-side only; decode must not rely on RNG).
- Stable softmax/logits to PMF:
  - Use log-sum-exp stabilization.
  - Apply fixed calibration (temperature/affine) logged in manifest.
- CDF quantization:
  - precision_bits (default 16)
  - epsilon floor for zero-prob symbols
  - monotone construction guarantees
- Validation:
  - Round-trip identity tests (hash original vs reconstructed).
  - NLL vs achieved bits tracked; target gap < 0.1 bpt.

8. Progressive development stages
- Stage 0 — Harness and manifest
  - Implement scaffolding, archive builder/reader, identity transform, CLI commands, and evaluate.
  - Outcome: deterministic pass-through; correctness and measurement.
- Stage 1 — Basic compression (training-free)
  - Option 1: lzma baseline (quickest viability).
  - Option 2: custom order-0 + rANS with 16-bit tables (more reusable).
  - Outcome: known baseline score; end-to-end pipeline proven.
- Stage 2 — Single reversible transform
  - Add temporal delta (mod 1024) per spatial location (or bit-plane coding as an alternative).
  - Keep order-0 model; measure delta.
- Stage 3 — Pretrained GPT probabilities (no training)
  - Wrap provided world model; fixed scan/context; single-parameter calibration.
  - Use rANS; quantize to 16-bit CDF.
  - Outcome: expected jump to ~2.6–2.8 score.
- Stage 4 — CDF and numerics hardening
  - Robust CDF building, tails, zero-prob handling; 2–4 way interleaving.
- Stage 5 — Second transform (optional)
  - Add bit-plane (if not in Stage 2), or vice versa; feed plane index as context feature.
- Stage 6 — Classic fallback & mixer (training-free)
  - Small PPM/CTW/Markov fallback; static logistic mixing with neural PMF.
- Stage 7 — Global adapter-based self-compression
  - Single LoRA/adapter trained once; quantized (~1–3 MB) and included once in archive.
- Stage 8 — Block-adaptive routing (small pool of adapters)
  - Fixed-size blocks choose from N tiny adapters; record only IDs.
- Stage 9 — Per-segment specialization with budget gating
  - Few gradient steps per segment; only include if net bytes saved > overhead.
- Stage 10 — Final polish
  - SIMD rANS, caching, checksums per stream, manifest evolution, profiling.

9. Evaluation methodology and metrics
- Correctness:
  - Bit-identical reconstruction across platforms (hash comparison).
  - Manifest version checks and integrity (SHA-256) on streams and side-info.
- Efficiency:
  - Total size (bytes), bits per token (bpt), challenge “score”.
  - Wall-clock compression and decompression times; memory peak.
- Reporting:
  - Store machine-readable reports (JSON) under reports/ with configs and results.
  - Track per-stage ablations and deltas.
- Benchmarks:
  - Fixed public shard for regression testing.
  - Include small/medium/large segment mixes.

10. Risk management
- Numerical drift/miscalibration:
  - Mitigate with deterministic calibration and CDF checks; strict CI tests.
- Adapter overhead exceeds savings:
  - Gate inclusion by net byte benefit; log keep/drop rationale.
- API creep / breakage:
  - Freeze module interfaces; evolve via manifest versioning.
- Decode performance:
  - Use interleaving and precomputed CDF tables; CPU profiling.

11. Security and integrity
- No network calls at decode time.
- Validate manifest version; reject unknown/unsafe parameters.
- Checksums for every bytestream and side-info file.
- Limit dependency footprint for decompressor (numpy, standard library).

12. Implementation layout (repo)
- compression/core/
  - io/: dataset_reader.py, archive_{builder,reader}.py, manifest.py
  - transforms/: identity.py, temporal_delta.py, bitplane.py
  - models/: order0.py, gpt_wrapper.py, classic_ppm.py, mixer.py
  - cdf/: quantize.py, calibrate.py
  - coder/: rans.py (or range.py), tables.py
  - cli/: compress.py, decompress.py, evaluate.py
  - tests/: roundtrip_test.py, determinism_test.py, cdf_test.py
- config/: stageN.yaml examples and defaults
- reports/: benchmark JSONs
- docs/: this architecture document and design notes

13. Compliance with competition rules
- Single-archive submission: archive contains all content needed to decode (manifest, streams, side-info).
- Single Python decompressor: compression/decompress.py reproduces exact tokens on CPU with documented CLI.
- No external downloads or training during decode; no randomness; deterministic behavior.

14. Glossary
- PMF: Probability mass function over the next symbol (0–1023).
- CDF: Cumulative distribution function used by entropy coders.
- rANS: Range Asymmetric Numeral Systems, an entropy coding method.
- Adapter/LoRA: Small set of trainable parameters added to a pretrained model to specialize it.

15. Change log
- v0.1 (Draft): Initial architecture, interfaces, CLIs, archive/manifest, staged roadmap.
- v0.2: Added canonical scan order, strict verification semantics, verification reporting flags.
