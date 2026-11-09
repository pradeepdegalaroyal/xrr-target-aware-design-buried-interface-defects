"""Parratt reflectivity forward model for a buried interface defect layer.

Stack, from the top: vacuum / niobium oxide cap / niobium film / defect layer / sapphire substrate.
Optical constants are generic electron-density estimates at 8.05 keV (illustrative, not fitted to data);
the absorption index is set to 5 percent of the dispersion decrement for every layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WAVELENGTH_A = 1.5406  # Angstrom (Cu K-alpha, about 8.05 keV)
K0 = 2.0 * np.pi / WAVELENGTH_A
DELTA_NB = 2.42e-5
DELTA_CAP = 1.35e-5
DELTA_SUB = 1.26e-5
BETA_RATIO = 0.05
CAP_THICKNESS_A = 20.0
CAP_ROUGHNESS_A = 4.0
SUB_ROUGHNESS_A = 3.0

PARAM_NAMES = ("t_defect_nm", "f_defect", "sigma_nm", "t_film_nm")
PRIOR_LOW = np.array([0.5, 0.40, 0.2, 24.5])
PRIOR_HIGH = np.array([3.0, 0.90, 0.6, 25.5])


@dataclass(frozen=True)
class Grid:
    """Momentum-transfer grid of candidate measurement settings (inverse Angstrom)."""

    q_min: float = 0.03
    q_max: float = 0.45
    n: int = 80

    def q(self) -> np.ndarray:
        return np.linspace(self.q_min, self.q_max, self.n)


def sample_prior(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw ``n`` parameter vectors uniformly from the prior box, shape (n, 4)."""
    return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * rng.random((n, 4))


def _kz(q: np.ndarray, delta: np.ndarray | float) -> np.ndarray:
    kz0 = q / 2.0
    return np.sqrt(kz0**2 - 2.0 * K0**2 * (delta - 1j * BETA_RATIO * delta) + 0j)


def reflectivity(theta: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Specular reflectivity R(q; theta) with Nevot-Croce roughness.

    Args:
        theta: array (n, 4) of (defect thickness nm, defect density fraction, defect interface
            roughness nm, film thickness nm).
        q: array (m,) of momentum transfers in inverse Angstrom.

    Returns:
        Array (n, m) of reflectivities.
    """
    theta = np.atleast_2d(theta)
    n = theta.shape[0]
    q = np.asarray(q, dtype=float)[None, :]
    t_d = theta[:, 0:1] * 10.0
    f_d = theta[:, 1:2]
    sig = theta[:, 2:3] * 10.0
    t_f = theta[:, 3:4] * 10.0
    kz = [
        _kz(q, 0.0) * np.ones((n, 1)),
        _kz(q, DELTA_CAP) * np.ones((n, 1)),
        _kz(q, DELTA_NB) * np.ones((n, 1)),
        _kz(q, f_d * DELTA_NB),
        _kz(q, DELTA_SUB) * np.ones((n, 1)),
    ]
    thick = [None, CAP_THICKNESS_A, t_f, t_d]
    rough = [CAP_ROUGHNESS_A, 3.0, sig, SUB_ROUGHNESS_A]
    x = np.zeros_like(kz[0])
    for j in range(3, -1, -1):
        r = (kz[j] - kz[j + 1]) / (kz[j] + kz[j + 1]) * np.exp(-2.0 * kz[j] * kz[j + 1] * rough[j] ** 2)
        if j == 3:
            x = r
        else:
            ph = np.exp(2j * kz[j + 1] * thick[j + 1])
            x = (r + x * ph) / (1.0 + r * x * ph)
    return np.abs(x) ** 2
