"""Amortised defect-thickness regressors: classical baselines and a data re-uploading quantum circuit.

The quantum model follows the linear-quantum-model view of data re-uploading circuits
(Jerbi et al., 2023): features are encoded several times between trainable rotation layers and a
trainable linear read-out of Pauli-Z expectations gives the prediction. It is simulated exactly
with a small batched statevector routine written in PyTorch.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor

from .physics import Grid, reflectivity, sample_prior

torch.set_num_threads(6)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")


def simulate_scan(
    rng: np.random.Generator, n: int, dose: float, bg: float, n_points: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``n`` uniform-scan datasets.

    Returns log-count features of shape (n, n_points) and defect-thickness labels in nm.
    """
    theta = sample_prior(rng, n)
    q = Grid().q()[:: 80 // n_points]
    mu = dose * reflectivity(theta, q) + bg
    y = rng.poisson(mu)
    return np.log1p(y), theta[:, 0]


# ----------------------------------------------------------------------------- statevector helpers
def _apply_1q(state: torch.Tensor, gate: torch.Tensor, wire: int, n: int) -> torch.Tensor:
    """Apply per-sample single-qubit gates ``gate`` (B, 2, 2) to ``wire`` of ``state`` (B, 2**n)."""
    b = state.shape[0]
    s = state.reshape(b, 2**wire, 2, 2 ** (n - wire - 1))
    s = torch.einsum("bij,bajc->baic", gate, s)
    return s.reshape(b, 2**n)


def _mat(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.stack([a, b], -1), torch.stack([c, d], -1)], -2)


def _rx(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    z = torch.zeros_like(c)
    return _mat(torch.complex(c, z), torch.complex(z, -s), torch.complex(z, -s), torch.complex(c, z))


def _ry(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    z = torch.zeros_like(c)
    return _mat(torch.complex(c, z), torch.complex(-s, z), torch.complex(s, z), torch.complex(c, z))


def _rz(a: torch.Tensor) -> torch.Tensor:
    z = torch.complex(torch.zeros_like(a), torch.zeros_like(a))
    return _mat(torch.complex(torch.cos(a / 2), -torch.sin(a / 2)), z, z, torch.complex(torch.cos(a / 2), torch.sin(a / 2)))


def _cz_ring(state: torch.Tensor, n: int) -> torch.Tensor:
    idx = torch.arange(2**n)
    sign = torch.ones(2**n)
    for w in range(n):
        a, b = w, (w + 1) % n
        bit_a = (idx >> (n - 1 - a)) & 1
        bit_b = (idx >> (n - 1 - b)) & 1
        sign = sign * (1 - 2 * (bit_a & bit_b).float())
    return state * sign.to(state.dtype)


class ReuploadingVQC(torch.nn.Module):
    """Data re-uploading variational quantum circuit (VQC) with a linear Pauli-Z read-out.

    n qubits, L layers of (encode, rotate, entangle). Parameters: rotations, encoding scale and
    offset, and the read-out weights.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 3, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n, self.L = n_qubits, n_layers
        self.rot = torch.nn.Parameter(torch.randn(n_layers, n_qubits, 3, generator=g) * 0.5)
        self.enc_w = torch.nn.Parameter(torch.ones(n_layers, n_qubits))
        self.enc_b = torch.nn.Parameter(torch.zeros(n_layers, n_qubits))
        self.out_w = torch.nn.Parameter(torch.randn(n_qubits, generator=g) * 0.5)
        self.out_b = torch.nn.Parameter(torch.zeros(1))

    def expectations(self, x: torch.Tensor) -> torch.Tensor:
        """Pauli-Z expectation of every qubit for inputs ``x`` of shape (B, n)."""
        b, n = x.shape[0], self.n
        state = torch.zeros(b, 2**n, dtype=torch.complex64)
        state[:, 0] = 1.0
        for layer in range(self.L):
            ang = self.enc_w[layer][None, :] * x + self.enc_b[layer][None, :]  # (B, n)
            r = self.rot[layer]  # (n, 3)
            rot_g = _rz(r[:, 0]) @ _ry(r[:, 1]) @ _rz(r[:, 2])  # (n, 2, 2)
            gates = rot_g[None] @ _rx(ang)  # (B, n, 2, 2)
            for w in range(n):
                state = _apply_1q(state, gates[:, w], w, n)
            state = _cz_ring(state, n)
        prob = state.real**2 + state.imag**2
        idx = torch.arange(2**n)
        z = torch.stack([1 - 2 * ((idx >> (n - 1 - w)) & 1).float() for w in range(n)], 0)
        return prob @ z.T

    def forward(self, x: torch.Tensor, shots: int | None = None, gen: torch.Generator | None = None) -> torch.Tensor:
        """Prediction; with ``shots`` set, add binomial-equivalent Gaussian read-out noise."""
        e = self.expectations(x)
        if shots is not None:
            sd = torch.sqrt(torch.clamp(1 - e**2, min=0) / shots)
            e = e + sd * torch.randn(e.shape, generator=gen)
        return e @ self.out_w + self.out_b


# ----------------------------------------------------------------------------- model fitting
def _prep(x_tr: np.ndarray, x_te: np.ndarray, n_comp: int) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_comp).fit(x_tr)
    z_tr, z_te = pca.transform(x_tr), pca.transform(x_te)
    s = z_tr.std(0) + 1e-9
    return z_tr / s, z_te / s


def fit_predict(
    model: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    seed: int,
    shots: int | None = None,
    n_layers: int = 3,
) -> np.ndarray:
    """Fit ``model`` on the training scans and return thickness predictions (nm) for ``x_te``.

    Models: ``ridge_full`` and ``mlp_full`` (all 40 features), ``mlp_small`` and ``mlp_large`` (multilayer
    perceptrons on 4 principal components) and ``vqc`` (4-qubit re-uploading circuit on the same
    4 components).
    """
    mu_y, sd_y = y_tr.mean(), y_tr.std()
    yt = (y_tr - mu_y) / sd_y
    if model == "ridge_full":
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(x_tr, yt)
        return m.predict(x_te) * sd_y + mu_y
    if model == "mlp_full":
        mu_x, sd_x = x_tr.mean(0), x_tr.std(0) + 1e-9
        m = MLPRegressor(hidden_layer_sizes=(32, 32), solver="lbfgs", max_iter=400, alpha=1e-2, random_state=seed).fit((x_tr - mu_x) / sd_x, yt)
        return m.predict((x_te - mu_x) / sd_x) * sd_y + mu_y
    z_tr, z_te = _prep(x_tr, x_te, 4)
    if model in ("mlp_small", "mlp_large"):
        hid = (8, 4) if model == "mlp_small" else (32, 32)
        m = MLPRegressor(hidden_layer_sizes=hid, solver="lbfgs", max_iter=400, alpha=1e-3, random_state=seed).fit(z_tr, yt)
        return m.predict(z_te) * sd_y + mu_y
    if model == "vqc":
        torch.manual_seed(seed)
        net = ReuploadingVQC(4, n_layers, seed)
        xt = torch.tensor(z_tr, dtype=torch.float32)
        yy = torch.tensor(yt, dtype=torch.float32)
        opt = torch.optim.Adam(net.parameters(), lr=0.05)
        for _ in range(100):
            opt.zero_grad()
            loss = torch.mean((net(xt) - yy) ** 2)
            loss.backward()
            opt.step()
        with torch.no_grad():
            gen = torch.Generator().manual_seed(seed + 99)
            p = net(torch.tensor(z_te, dtype=torch.float32), shots=shots, gen=gen).numpy()
        return p * sd_y + mu_y
    raise ValueError(model)
