"""
Archive I/O utilities for Stage 0 (streaming, memory-safe).

Provides:
- ArchiveBuilder: create a single-archive (zip) with a manifest and arbitrary side-info / streams.
  Streams are written immediately to disk instead of being buffered in memory.
- ArchiveReader: read manifest and extract streams for deterministic decompression.

The manifest schema is minimal for Stage 0 and will be extended later.
"""
import json
import zipfile
import hashlib
import tempfile
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ARCHIVE_MANIFEST_NAME = "manifest.json"


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


class ArchiveBuilder:
    """
    Memory-safe archive builder that streams each added file into a temporary zip
    file on disk immediately instead of buffering bytes in RAM.

    Usage:
      b = ArchiveBuilder(out_path, verbose=True)
      b.add_stream("data/segment-000.npy", bytes_data)
      b.set_manifest(manifest_dict)
      b.finalize()
    """

    def __init__(self, out_path: str, verbose: bool = False):
        self.out_path = Path(out_path).resolve()
        # create a temp directory and a temporary zip file path inside it
        self._tmpdir = Path(tempfile.mkdtemp(prefix="commavq_archive_"))
        self._tmpzip = self._tmpdir / "archive.zip"
        # open zip for writing; keep it open so add_stream can write immediately
        self._zip = zipfile.ZipFile(self._tmpzip, "w", compression=zipfile.ZIP_DEFLATED)
        self._manifest: Optional[Dict] = None
        # metadata for streams (path, size, sha256)
        self._streams_meta: List[Dict] = []
        # counters
        self.bytes_written: int = 0
        self.files_written: int = 0
        self.verbose = bool(verbose)
        self._logger = logging.getLogger("compression.io")
        if self.verbose:
            self._logger.info(f"ArchiveBuilder initialized (tmp: {self._tmpzip})")

    def add_stream(self, relpath: str, data: bytes) -> None:
        """
        Stream a bytes payload into the zip under relpath (posix path inside archive).
        Records sha256 and size in streams meta.
        """
        rel = Path(relpath)
        rel_posix = rel.as_posix()
        # write immediately to the open zipfile
        try:
            self._zip.writestr(rel_posix, data)
        except Exception:
            # On some Windows platforms writing large zip entries can fail;
            # re-raise as a clearer error.
            raise
        meta = {"path": rel_posix, "size_bytes": len(data), "sha256": _sha256_bytes(data)}
        self._streams_meta.append(meta)
        self.files_written += 1
        self.bytes_written += len(data)
        if self.verbose:
            self._logger.info(f"wrote stream: {rel_posix} ({len(data)} bytes)")

    def set_manifest(self, manifest: Dict) -> None:
        """
        Set the manifest dict. We'll augment it with stream metadata automatically
        in finalize() and then write manifest.json into the archive.
        """
        self._manifest = dict(manifest)  # shallow copy
        if self.verbose:
            self._logger.info("manifest set (will be written on finalize)")

    def finalize(self) -> None:
        """
        Write manifest + close zip archive and move the temporary zip into place.
        Overwrites existing file at self.out_path.
        """
        if self._manifest is None:
            self._manifest = {}
        # attach streams meta
        self._manifest.setdefault("integrity", {})
        self._manifest["integrity"]["streams"] = self._streams_meta

        # write manifest into the open zip
        manifest_bytes = json.dumps(self._manifest, indent=2).encode("utf-8")
        # ensure manifest doesn't already exist in the zip (it shouldn't)
        try:
            # writing manifest last
            self._zip.writestr(ARCHIVE_MANIFEST_NAME, manifest_bytes)
        finally:
            # always close zip to flush data
            try:
                self._zip.close()
            except Exception:
                pass

        # ensure parent exists
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        # atomically replace if exists
        if self.out_path.exists():
            self.out_path.unlink()
        # move tmpzip into final location
        self._tmpzip.replace(self.out_path)
        # cleanup temp directory (attempt)
        try:
            self._tmpdir.rmdir()
        except Exception:
            # leave it if not empty for debugging
            pass
        if self.verbose:
            size = self.out_path.stat().st_size if self.out_path.exists() else 0
            self._logger.info(f"finalized archive: {self.out_path} ({self.files_written} files, {self.bytes_written} bytes, {size} bytes on disk)")

    def cleanup(self) -> None:
        # Close zip if still open and try to remove tmpdir
        try:
            if hasattr(self, "_zip") and getattr(self._zip, "fp", None) is not None:
                try:
                    self._zip.close()
                except Exception:
                    pass
            # remove any remaining files in the tmpdir if present
            for p in self._tmpdir.iterdir():
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                self._tmpdir.rmdir()
            except Exception:
                pass
        except Exception:
            pass


class ArchiveReader:
    """
    Read the archive created by ArchiveBuilder.
    Usage:
      r = ArchiveReader("archive.zip")
      manifest = r.manifest()
      r.extract_all(out_dir)   # or r.read_stream_bytes("data/x.npy")
    """

    def __init__(self, archive_path: str):
        self.archive_path = Path(archive_path).resolve()
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {self.archive_path}")
        self._zip = zipfile.ZipFile(self.archive_path, "r")
        self._manifest = None

    def manifest(self) -> Dict:
        if self._manifest is None:
            try:
                manifest_bytes = self._zip.read(ARCHIVE_MANIFEST_NAME)
                self._manifest = json.loads(manifest_bytes.decode("utf-8"))
            except KeyError:
                raise RuntimeError("manifest.json not found in archive")
        return self._manifest

    def list_streams(self) -> List[str]:
        return [zi.filename for zi in self._zip.infolist() if zi.filename != ARCHIVE_MANIFEST_NAME]

    def read_stream_bytes(self, relpath: str) -> bytes:
        return self._zip.read(relpath)

    def extract_all(self, out_dir: str) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Extract manifest as well for debugging
        for zi in self._zip.infolist():
            target = out_dir / zi.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                f.write(self._zip.read(zi.filename))

    def close(self) -> None:
        try:
            self._zip.close()
        except Exception:
            pass
