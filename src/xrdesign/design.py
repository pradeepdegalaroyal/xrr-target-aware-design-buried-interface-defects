"""Sequential Bayesian design: importance-weighted posterior and measurement-selection rules."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .physics import PRIOR_HIGH, PRIOR_LOW, Grid, reflectivity, sample_prior

METHODS = ("uniform", "random", "uncertainty", "all_parameter", "target_aware")
T_CRIT_NM = 1.5  # screening threshold on the defect-layer thickness (nm)


def van_der_corput_order(n: int) -> np.ndarray:
    """Space-filling visiting order of ``n`` grid points (bit-reversal permutation)."""
    bits = int(np.ceil(np.log2(n)))
    idx = [int(format(i, f"0{bits}b")[::-1], 2) for i in range(2**bits)]
    return np.array([i for i in idx if i < n])


@dataclass
class Library:
    """Prior particles with cached reflectivity on the candidate grid."""

    theta: np.ndarray  # (N, 4)
    refl: np.ndarray  # (N, Q) float32
    q: np.ndarray

    @classmethod
    def build(cls, rng: np.random.Generator, n: int, grid: Grid, chunk: int = 10000) -> "Library":
        theta = sample_prior(rng, n)
        q = grid.q()
        refl = np.concatenate([reflectivity(theta[i : i + chunk], q) for i in range(0, n, chunk)]).astype(np.float32)
        return cls(theta=theta, refl=refl, q=q)


@dataclass
class Posterior:
    """Importance-weighted posterior over the library particles."""

    lib: Library
    dose: float
    bg_assumed: float
    logw: np.ndarray = field(init=False)
    mu: np.ndarray = field(init=False)  # expected counts (N, Q), float32
    mu2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.logw = np.zeros(len(self.lib.theta))
        self.mu = (self.dose * self.lib.refl + self.bg_assumed).astype(np.float32)
        self.mu2 = self.mu**2

    def update(self, j: int, y: int) -> None:
        mu = self.mu[:, j].astype(np.float64)
        self.logw += y * np.log(mu) - mu

    def weights(self) -> np.ndarray:
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def ess(self) -> float:
        w = self.weights()
        return float(1.0 / np.sum(w**2))

    def mean_target(self) -> float:
        return float(self.weights() @ self.lib.theta[:, 0])

    def interval_target(self, level: float = 0.9) -> tuple[float, float, float]:
        """Weighted credible interval of the defect thickness and P(thickness > T_CRIT_NM)."""
        w = self.weights()
        t = self.lib.theta[:, 0]
        o = np.argsort(t)
        c = np.cumsum(w[o])
        lo = t[o][np.searchsorted(c, (1 - level) / 2)]
        hi = t[o][min(np.searchsorted(c, 1 - (1 - level) / 2), len(t) - 1)]
        return float(lo), float(hi), float(w[t > T_CRIT_NM].sum())


def _score_lmmse(post: Posterior, params: list[int]) -> np.ndarray:
    """Expected fractional variance reduction of the chosen parameters per candidate setting.

    Linear minimum-mean-square-error estimate of each parameter from one Poisson count, evaluated
    under the current posterior (nonlinear in theta through the cached reflectivity).
    """
    w = post.weights().astype(np.float32)
    m = (w @ post.mu).astype(np.float64)
    var_mu = np.maximum((w @ post.mu2).astype(np.float64) - m**2, 0.0)
    th = post.lib.theta[:, params].astype(np.float32)
    tc = th - w @ th
    var_t = (w[:, None] * tc**2).sum(0).astype(np.float64)
    cov = ((w[:, None] * tc).T @ post.mu).astype(np.float64)  # (P, Q)
    return np.mean(cov**2 / (var_t[:, None] * (var_mu + m)[None, :]), axis=0)


def select(method: str, post: Posterior, step: int, order: np.ndarray, rng: np.random.Generator) -> int:
    """Return the index of the next candidate setting under the named rule."""
    q_n = len(post.lib.q)
    if method == "uniform":
        return int(order[step % q_n])
    if method == "random":
        return int(rng.integers(q_n))
    if method == "uncertainty":
        w = post.weights().astype(np.float32)
        m = (w @ post.mu).astype(np.float64)
        return int(np.argmax(np.maximum((w @ post.mu2).astype(np.float64) - m**2, 0.0) / m))
    if method == "all_parameter":
        return int(np.argmax(_score_lmmse(post, [0, 1, 2, 3])))
    if method == "target_aware":
        return int(np.argmax(_score_lmmse(post, [0])))
    raise ValueError(method)


def run_experiment(
    method: str,
    lib: Library,
    theta_true: np.ndarray,
    steps: int,
    dose: float,
    bg_true: float,
    bg_assumed: float,
    rng: np.random.Generator,
    record_every: int = 4,
) -> list[dict]:
    """Run one sequential experiment; return posterior summaries at checkpoints."""
    r_true = reflectivity(theta_true[None, :], lib.q)[0]
    post = Posterior(lib, dose, bg_assumed)
    order = van_der_corput_order(len(lib.q))
    rows: list[dict] = []
    for s in range(steps):
        j = select(method, post, s, order, rng)
        y = rng.poisson(dose * r_true[j] + bg_true)
        post.update(j, int(y))
        if (s + 1) % record_every == 0 or s + 1 == steps:
            lo, hi, p_bad = post.interval_target()
            t_true = float(theta_true[0])
            rows.append(
                dict(
                    method=method,
                    step=s + 1,
                    photons=(s + 1) * dose,
                    t_true=t_true,
                    t_hat=post.mean_target(),
                    err=post.mean_target() - t_true,
                    width90=hi - lo,
                    covered=int(lo <= t_true <= hi),
                    decision_correct=int((p_bad > 0.5) == (t_true > T_CRIT_NM)),
                    ess=post.ess(),
                )
            )
    return rows
