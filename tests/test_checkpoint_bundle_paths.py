from pathlib import Path

import pytest
from omegaconf import OmegaConf

from prism.train.utils import bundle_checkpoint


def _create_run(path):
    for step in ("best/12", "10", "20"):
        checkpoint = path / "checkpoints" / step
        checkpoint.mkdir(parents=True)
        (checkpoint / "data").write_bytes(f"checkpoint {step}".encode())
    OmegaConf.save(
        {
            "name": "model",
            "model": {"name": "ExampleModel"},
            "simulator": {"sim_type": {"name": "Ball3StickNoise"}},
        },
        path / "config.yaml",
    )


def _snapshot(root):
    entries = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            entry = ("symlink", str(path.readlink()))
        elif path.is_file():
            entry = ("file", path.read_bytes())
        else:
            entry = ("directory", None)
        entries[str(path.relative_to(root))] = entry
    return entries


@pytest.mark.parametrize(
    "relation", ["same", "ancestor", "descendant", "new_descendant"]
)
@pytest.mark.parametrize("alias", [None, "source", "output_parent", "destination"])
def test_overlapping_bundle_paths_do_not_modify_source(tmp_path, relation, alias):
    run = tmp_path / "runs" / "model" / "run"
    _create_run(run)
    destination = {
        "same": run,
        "ancestor": run.parent,
        "descendant": run / "exports" / "model",
        "new_descendant": run / "new" / "model",
    }[relation]
    if relation != "new_descendant":
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "keep").write_text("Existing destination contents")
    output_dir, model_name = destination.parent, destination.name
    source = run
    if alias == "source":
        source = tmp_path / "run-link"
        source.symlink_to(run, target_is_directory=True)
    elif alias == "output_parent":
        output_dir = tmp_path / "output-link"
        output_dir.symlink_to(destination.parent, target_is_directory=True)
    elif alias == "destination":
        output_dir = tmp_path / "bundles"
        output_dir.mkdir()
        (output_dir / model_name).symlink_to(destination, target_is_directory=True)
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="must not overlap"):
        bundle_checkpoint(source, output_dir, model_name=model_name, which="best")

    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("relation", ["same", "ancestor", "descendant"])
def test_bundle_rejects_overlap_with_external_checkpoint_target(tmp_path, relation):
    run = tmp_path / "run"
    _create_run(run)
    external_checkpoint = tmp_path / "archive" / "best"
    external_checkpoint.parent.mkdir()
    (run / "checkpoints" / "best").rename(external_checkpoint)
    (run / "checkpoints" / "best").symlink_to(
        external_checkpoint, target_is_directory=True
    )
    destination = {
        "same": external_checkpoint,
        "ancestor": external_checkpoint.parent,
        "descendant": external_checkpoint / "bundle",
    }[relation]
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="must not overlap"):
        bundle_checkpoint(
            run, destination.parent, model_name=destination.name, which="best"
        )

    assert _snapshot(tmp_path) == before


def test_bundle_rejects_parent_directory_model_name(tmp_path):
    run = tmp_path / "run"
    _create_run(run)
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="single directory name"):
        bundle_checkpoint(run, tmp_path / "bundles", model_name="..", which="best")

    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("which", "expected_steps"),
    [
        ("best", {"best"}),
        ("latest", {"20"}),
        ("best_and_latest", {"best", "20"}),
        ("all", {"best", "10", "20"}),
        (10, {"10"}),
    ],
)
def test_bundle_replaces_disjoint_destination_and_preserves_source(
    tmp_path, which, expected_steps
):
    run = tmp_path / "run"
    _create_run(run)
    before = _snapshot(run)
    output_dir = tmp_path / "bundles"
    destination = output_dir / "model"
    destination.mkdir(parents=True)
    (destination / "stale").write_text("Previous bundle contents")

    result = bundle_checkpoint(run, output_dir, which=which)

    assert result == destination
    assert _snapshot(run) == before
    assert not (destination / "stale").exists()
    assert (destination / "config.yaml").is_file()
    assert (destination / "artifact.yaml").is_file()
    assert {
        path.name for path in (destination / "checkpoints").iterdir()
    } == expected_steps
    for step in expected_steps:
        relative = Path("checkpoints") / step
        assert _snapshot(destination / relative) == _snapshot(run / relative)
