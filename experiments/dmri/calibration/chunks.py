"""Resume verified calibration chunks and publish their summaries."""

import json
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from experiments.dmri._shared.metrics import summarize, validate_batch
from experiments.dmri._shared.plotting import plots
from experiments.dmri._shared.storage import (
    canonical_hash,
    file_manifest,
    file_sha256,
    write_json,
)

BASE_FIELDS = {
    "trial_index",
    "model_mask",
    "theta_true",
    "mask_prior",
    "num_acquisitions",
}
MODEL_FIELDS = BASE_FIELDS | {"model_pit"}
THETA_FIELDS = BASE_FIELDS | {"theta_sum_pit", "theta_pit", "theta_mask"}
CHUNK_NAME = re.compile(r"([0-9]{7})-([0-9]{7})")


def write_new_json(path, value):
    """Never replace an existing protocol, completion record, or attempt log."""
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_output(path, protocol, *, resume):
    path = Path(path)
    if path.exists() and not resume:
        raise FileExistsError(
            f"Output exists; use --resume to verify and continue: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)
    protocol_path = path / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Resume protocol differs from the existing evaluation")
    else:
        # A crash before protocol creation may leave only the process lock file.
        if any(p.name != ".lock" for p in path.iterdir()):
            raise ValueError("Existing evaluation files lack their immutable protocol")
        write_new_json(protocol_path, protocol)
    return canonical_hash(protocol)


def validate_values(values, start, stop):
    if set(values) not in (MODEL_FIELDS, THETA_FIELDS):
        raise ValueError("SBC output fields differ from the required schema")
    if any(array.ndim == 0 or len(array) != stop - start for array in values.values()):
        raise ValueError("SBC arrays do not contain exactly the requested trial range")
    if not np.array_equal(values["trial_index"], np.arange(start, stop)):
        raise ValueError("Trial indices contain gaps, duplicates, or a different range")
    validate_batch(values)
    for field in ("model_pit", "theta_sum_pit", "theta_pit"):
        if field in values:
            active = (
                values[field][values["theta_mask"].astype(bool)]
                if field == "theta_pit"
                else values[field]
            )
            if np.any((active < 0) | (active > 1)):
                raise ValueError(f"SBC {field} lies outside [0, 1]")


def commit_chunk(directory, values, *, start, stop, sequence, protocol_hash, seconds):
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{start:07d}-{stop:07d}"
    if output.exists():
        raise FileExistsError(f"Completed trial chunk already exists: {output}")
    validate_values(values, start, stop)
    temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=directory))
    try:
        np.savez_compressed(temporary / "samples.npz", **values)
        marker = {
            "version": 1,
            "start": start,
            "stop": stop,
            "trials": stop - start,
            "sequence_index": sequence,
            "protocol_sha256": protocol_hash,
            "samples_sha256": file_sha256(temporary / "samples.npz"),
            "compute_seconds": seconds,
        }
        write_new_json(temporary / "complete.json", marker)
        # The caller holds an exclusive run lock; rename publishes the whole chunk.
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return marker


def load_chunks(directory, *, trials, chunk_size, sequence, protocol_hash):
    """Read only committed chunks; incomplete temporary directories are ignored."""
    if not directory.exists():
        return [], []
    arrays, records, expected_start = [], [], 0
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".writing-"):
            continue
        match = CHUNK_NAME.fullmatch(path.name)
        if not match or not path.is_dir() or path.is_symlink():
            raise ValueError(f"Unrecognized SBC chunk path: {path}")
        start, stop = map(int, match.groups())
        if start != expected_start or stop != min(start + chunk_size, trials):
            raise ValueError(
                "SBC chunks are not contiguous ranges of the requested size"
            )
        marker = json.loads((path / "complete.json").read_text())
        expected = {
            "version": 1,
            "start": start,
            "stop": stop,
            "trials": stop - start,
            "sequence_index": sequence,
            "protocol_sha256": protocol_hash,
        }
        if any(marker.get(key) != value for key, value in expected.items()):
            raise ValueError("Chunk completion metadata differs from this evaluation")
        if file_sha256(path / "samples.npz") != marker["samples_sha256"]:
            raise ValueError("Completed SBC chunk content hash changed")
        with np.load(path / "samples.npz", allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        validate_values(values, start, stop)
        arrays.append(values)
        records.append(marker)
        expected_start = stop
    return arrays, records


def evaluate_sequence(
    output,
    compute,
    *,
    trials,
    batch_size,
    chunk_size,
    sequence,
    protocol_hash,
    on_chunk=None,
):
    """Continue from the first missing trial, recomputing no committed chunk."""
    output.mkdir(parents=True, exist_ok=True)
    directory = output / "chunks"
    chunks, records = load_chunks(
        directory,
        trials=trials,
        chunk_size=chunk_size,
        sequence=sequence,
        protocol_hash=protocol_hash,
    )
    completed = records[-1]["stop"] if records else 0
    for start in range(completed, trials, chunk_size):
        stop = min(start + chunk_size, trials)
        started, batches = time.monotonic(), []
        for batch_start in range(start, stop, batch_size):
            batch_started = time.monotonic()
            count = min(batch_size, stop - batch_start)
            indices = jnp.arange(
                batch_start, batch_start + batch_size, dtype=jnp.uint32
            )
            values = {
                key: np.asarray(value)[:count]
                for key, value in compute(indices).items()
            }
            try:
                validate_values(values, batch_start, batch_start + count)
            except (ValueError, FloatingPointError):
                failures = output / "failures"
                failures.mkdir(exist_ok=True)
                np.savez_compressed(
                    failures / f"batch-{batch_start:07d}-{uuid.uuid4().hex}.npz",
                    **values,
                )
                raise
            batches.append(values)
            if batch_start < completed + 2 * batch_size:
                print(
                    json.dumps({
                        "event": "batch_timing",
                        "dataset": output.name,
                        "sequence_index": sequence,
                        "trial_start": batch_start,
                        "trials": count,
                        "seconds": time.monotonic() - batch_started,
                        "includes_compilation": batch_start == completed,
                    }),
                    flush=True,
                )
        values = {
            key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]
        }
        record = commit_chunk(
            directory,
            values,
            start=start,
            stop=stop,
            sequence=sequence,
            protocol_hash=protocol_hash,
            seconds=time.monotonic() - started,
        )
        chunks.append(values)
        records.append(record)
        progress = {
            "sequence_index": sequence,
            "completed_trials": stop,
            "requested_trials": trials,
            "last_chunk_seconds": record["compute_seconds"],
        }
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        if on_chunk:
            on_chunk(progress)
    if not records or records[-1]["stop"] != trials:
        raise ValueError("Sequence did not finish exactly the requested trial count")
    arrays = {
        key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]
    }
    validate_values(arrays, 0, trials)
    return arrays, records


def finalize_sequence(output, arrays, records, *, sequence, protocol_hash, label):
    """Publish summaries and plots together, without replacing earlier results."""
    final = output / "summary"
    identity = {
        "protocol_sha256": protocol_hash,
        "sequence_index": sequence,
        "completed_trials": len(arrays["trial_index"]),
        "chunks_sha256": canonical_hash(records),
    }
    if final.exists():
        marker = json.loads((final / "complete.json").read_text())
        if any(marker.get(key) != value for key, value in identity.items()):
            raise ValueError(
                "Completed sequence summary differs from its verified chunks"
            )
        files = file_manifest(final)
        files.pop("complete.json")
        if files != marker["files"]:
            raise ValueError("Completed sequence summary files changed")
        return json.loads((final / "summary.json").read_text())
    temporary = Path(tempfile.mkdtemp(prefix=".summary-", dir=output))
    try:
        np.savez_compressed(temporary / "samples.npz", **arrays)
        summary = summarize(arrays)
        acquisition_counts, count_frequencies = np.unique(
            arrays["num_acquisitions"], return_counts=True
        )
        summary["observed_acquisition_counts"] = {
            str(int(count)): int(frequency)
            for count, frequency in zip(
                acquisition_counts, count_frequencies, strict=True
            )
        }
        summary.update({
            **identity,
            "status": "completed",
            "samples_sha256": file_sha256(temporary / "samples.npz"),
            "chunk_count": len(records),
            "compute_seconds": sum(record["compute_seconds"] for record in records),
            "exact_trial_range": [0, len(arrays["trial_index"])],
        })
        plots(arrays, summary, temporary, label)
        write_new_json(temporary / "summary.json", summary)
        write_new_json(
            temporary / "complete.json", {**identity, "files": file_manifest(temporary)}
        )
        temporary.rename(final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return summary
