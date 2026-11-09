"""Tests for the statevector re-uploading circuit and the regressors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pennylane as qml
import torch

from xrdesign.amortised import ReuploadingVQC, fit_predict, simulate_scan


def test_vqc_matches_pennylane():
    net = ReuploadingVQC(3, 2, seed=1)
    x = torch.randn(2, 3)
    ours = net.expectations(x).detach().numpy()
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(xx):
        for layer in range(2):
            for w in range(3):
                qml.RX(float(net.enc_w[layer, w] * xx[w] + net.enc_b[layer, w]), wires=w)
                r = net.rot[layer, w].detach().numpy()
                qml.RZ(r[2], wires=w)
                qml.RY(r[1], wires=w)
                qml.RZ(r[0], wires=w)
            for w in range(3):
                qml.CZ(wires=[w, (w + 1) % 3])
        return [qml.expval(qml.PauliZ(w)) for w in range(3)]

    ref = np.array([circuit(x[i].numpy()) for i in range(2)])
    assert np.abs(ours - ref).max() < 1e-4


def test_regressors_run_and_vqc_shots_add_noise():
    rng = np.random.default_rng(0)
    x, y = simulate_scan(rng, 60, 1e4, 1.0)
    for model in ("ridge_full", "mlp_small", "vqc"):
        p = fit_predict(model, x[:40], y[:40], x[40:], seed=0, n_layers=1)
        assert p.shape == (20,) and np.isfinite(p).all()
    a = fit_predict("vqc", x[:40], y[:40], x[40:], seed=0, n_layers=1)
    b = fit_predict("vqc", x[:40], y[:40], x[40:], seed=0, n_layers=1, shots=20)
    assert not np.allclose(a, b)
