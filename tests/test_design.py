"""Tests for the reflectivity model and the sequential design loop."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from xrdesign.design import METHODS, Library, run_experiment, van_der_corput_order
from xrdesign.physics import Grid, reflectivity, sample_prior


def test_reflectivity_bounds_and_total_reflection():
    th = sample_prior(np.random.default_rng(0), 50)
    r = reflectivity(th, Grid().q())
    assert np.all(r >= 0) and np.all(r <= 1.0 + 1e-9)
    assert np.all(r[:, 0] > 0.5)  # near the critical edge reflectivity is high
    assert np.all(r[:, -1] < r[:, 0])


def test_defect_thickness_changes_curve():
    a = np.array([[0.6, 0.6, 0.4, 25.0]])
    b = np.array([[2.8, 0.6, 0.4, 25.0]])
    q = Grid().q()
    assert np.max(np.abs(np.log(reflectivity(a, q)) - np.log(reflectivity(b, q)))) > 0.05


def test_van_der_corput_is_permutation():
    o = van_der_corput_order(80)
    assert sorted(o.tolist()) == list(range(80))


def test_all_methods_run_and_return_finite_posterior():
    rng = np.random.default_rng(0)
    lib = Library.build(rng, 3000, Grid(n=32))
    truth = sample_prior(rng, 1)[0]
    for m in METHODS:
        rows = run_experiment(m, lib, truth, 8, 1e4, 1.0, 1.0, np.random.default_rng(1), record_every=4)
        assert len(rows) == 2 and np.isfinite(rows[-1]["t_hat"])
        assert 0.5 <= rows[-1]["t_hat"] <= 3.0
