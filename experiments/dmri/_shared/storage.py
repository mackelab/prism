"""File hashes, atomic JSON output, and source records for dMRI experiments."""

import hashlib
import json
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(root: Path, *, allow_file_symlinks=False) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and not (allow_file_symlinks and path.is_file()):
            raise ValueError(f"A snapshot must not follow a symlink: {path}")
        if path.is_file():
            files[str(path.relative_to(root))] = file_sha256(path)
    return files


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def source_provenance(source):
    """Record the loaded library checkout without requiring a historical HEAD."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": None}
    return {"commit": commit, "dirty": bool(status), "status": status.splitlines()}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
