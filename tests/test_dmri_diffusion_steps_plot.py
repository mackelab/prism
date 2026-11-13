import numpy as np
import pytest

from experiments.dmri.diffusion_steps import plot


def test_summarize_and_figure(tmp_path):
    steps = np.array([2, 4, 8])
    ess = np.array([[0.1, 0.3], [0.2, 0.4], [0.5, 0.5]])
    mean, low, high = plot.summarize(ess)
    np.testing.assert_allclose(mean, [0.2, 0.3, 0.5])
    assert (low <= mean).all() and (mean <= high).all()
    assert len(plot.plot_steps(steps, ess, ess / 100).axes) == 2
    results = tmp_path / "r.npz"
    np.savez(results, num_steps=steps, ess=ess, run_time_s=ess / 100)
    out = tmp_path / "f.svg"
    plot.main([str(results), "--output", str(out)])
    assert out.exists()
    with pytest.raises(FileExistsError):
        plot.main([str(results), "--output", str(out)])
