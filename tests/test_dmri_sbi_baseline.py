"""The torch neural-spline-flow baseline (optional ``torch`` and ``sbi``)."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sbi")

from experiments.dmri.sbi_baseline import evaluate, train  # noqa: E402
from experiments.dmri.sbi_baseline.model import (  # noqa: E402
    DMRIEncoder,
    build_flow,
    negative_log_likelihood,
    pack_condition,
    unpack_condition,
)
from prism.dmri.simulators.models import Ball3StickSharedDiffusivity  # noqa: E402

TINY = [
    "--model-dim",
    "16",
    "--output-dim",
    "16",
    "--num-cls-tokens",
    "2",
    "--num-layers",
    "1",
    "--hidden-features",
    "16",
    "--num-transforms",
    "2",
]


def condition(batch, measurements):
    generator = torch.Generator().manual_seed(measurements)
    return pack_condition(
        torch.rand(batch, measurements, generator=generator),
        torch.rand(batch, measurements, generator=generator) * 4000,
        torch.randn(batch, measurements, 3, generator=generator),
    )


def test_condition_packing_round_trips():
    signal, bvals, bvecs = torch.rand(3, 7), torch.rand(3, 7), torch.randn(3, 7, 3)
    for original, restored in zip(
        (signal, bvals, bvecs),
        unpack_condition(pack_condition(signal, bvals, bvecs)),
        strict=True,
    ):
        assert torch.equal(original, restored)
    with pytest.raises(ValueError, match="multiple of 5"):
        unpack_condition(torch.zeros(2, 12))


def test_encoder_accepts_any_number_of_measurements():
    encoder = DMRIEncoder(model_dim=16, output_dim=8, num_cls_tokens=2, num_layers=1)
    assert encoder(condition(4, 105)).shape == (4, 8)
    assert encoder(condition(4, 297)).shape == (4, 8)


def test_one_flow_serves_both_acquisition_lengths():
    theta = torch.randn(6, 11)
    flow = build_flow(
        theta,
        condition(6, 105),
        model_dim=16,
        output_dim=16,
        num_cls_tokens=2,
        num_layers=1,
        hidden_features=16,
        num_transforms=2,
    )
    for measurements in (105, 297):
        cond = condition(6, measurements)
        loss = negative_log_likelihood(flow, theta, cond)
        assert loss.shape == (6,) and torch.isfinite(loss).all()
        assert flow.sample(5, context=cond).shape == (6, 5, 11)
    negative_log_likelihood(flow, theta, condition(6, 105)).mean().backward()
    assert all(p.grad is not None for p in flow.parameters() if p.requires_grad)


@pytest.mark.parametrize("large, measurements", [(False, 105), (True, 297)])
def test_simulated_batches_have_the_acquisition_shapes(large, measurements):
    sim_type = Ball3StickSharedDiffusivity
    batch = train.batch_function(sim_type, large_acquisition=large, batch_size=3)(
        jax.random.PRNGKey(0)
    )
    theta, signal, bvals, bvecs = (np.asarray(a) for a in batch)
    assert theta.shape == (3, sim_type.theta_dim)
    assert signal.shape == bvals.shape == (3, measurements) and bvecs.shape == (
        3,
        measurements,
        3,
    )
    assert np.isfinite(signal).all()


def test_effective_sample_fraction():
    n = 8
    assert float(
        evaluate.effective_sample_fraction(jnp.zeros(n), jnp.zeros(n))
    ) == pytest.approx(1.0)
    dominant = jnp.zeros(n).at[0].set(50.0)
    assert float(
        evaluate.effective_sample_fraction(dominant, jnp.zeros(n))
    ) == pytest.approx(1 / n, rel=1e-3)
    with_invalid = jnp.zeros(n).at[3].set(-jnp.inf)
    assert float(
        evaluate.effective_sample_fraction(with_invalid, jnp.zeros(n))
    ) == pytest.approx((n - 1) / n)


def test_train_resume_and_evaluate(tmp_path):
    run = tmp_path / "run"
    common = [
        "--output",
        str(run),
        "--batch-size",
        "4",
        "--eval-batch-size",
        "4",
        "--eval-every",
        "1",
        "--checkpoint-every",
        "1",
        "--device",
        "cpu",
        *TINY,
    ]
    train.main([*common, "--num-steps", "2"])
    latest = torch.load(run / "checkpoints" / "latest.pt", map_location="cpu")
    assert latest["step"] == 2 and (run / "checkpoints" / "best.pt").exists()
    assert (
        json.loads((run / "config.json").read_text())["sim_type"]
        == "Ball3StickSharedDiffusivity"
    )
    train.main([*common, "--num-steps", "3", "--resume"])
    assert (
        torch.load(run / "checkpoints" / "latest.pt", map_location="cpu")["step"] == 3
    )

    output = tmp_path / "metrics.json"
    evaluate.main([
        str(run / "checkpoints" / "best.pt"),
        "--regimes",
        "hcp",
        "--num-datasets",
        "2",
        "--num-samples",
        "8",
        "--device",
        "cpu",
        "--output",
        str(output),
    ])
    metrics = json.loads(output.read_text())["hcp"]
    assert set(metrics) == {"ess", "rmse", "ksd", "ksd_pvalue"}
    assert 0.0 <= metrics["ess"]["mean"] <= 1.0 and metrics["rmse"]["mean"] > 0
    with pytest.raises(FileExistsError):
        evaluate.main([str(run / "checkpoints" / "best.pt"), "--output", str(output)])
