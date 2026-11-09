# Method

## Forward model

A Parratt recursion with Nevot-Croce roughness gives the specular reflectivity R(q; theta) of a stack (top to bottom): vacuum, niobium-oxide cap (2.0 nm, fixed), niobium film, buried defect layer, sapphire substrate. Optical constants are generic electron-density estimates at 8.05 keV; the absorption index is 5 % of the dispersion decrement in every layer. They are illustrative, not fitted. The unknowns theta are the defect thickness (0.5 to 3.0 nm, the target), the defect density as a fraction of niobium (0.40 to 0.90), the roughness of the film/defect interface (0.2 to 0.6 nm) and the film thickness (24.5 to 25.5 nm), all with uniform priors. The candidate settings are 80 values of momentum transfer q between 0.03 and 0.45 inverse angstrom.

## Measurement and inference

Each step spends a fixed dose of 3 x 10^4 incident photons at one chosen q and records a Poisson count with mean dose x R + background. The posterior over theta is an importance-weighted library of 50 000 prior particles whose reflectivity curves are cached; each count multiplies the weights by the Poisson likelihood (the analyst assumes a background of 1 count per step). The effective sample size (ESS) is logged for every run.

## Selection rules (the compared methods)

1. **Uniform scan** (classical baseline): settings visited in a space-filling (bit-reversal) order, so any prefix of the scan covers the whole q range.
2. **Random setting**: a setting drawn uniformly at each step.
3. **Predictive-variance sampling**: the setting with the largest ratio of predictive variance to predictive mean under the current posterior (the usual uncertainty-sampling rule from active learning).
4. **All-parameter design** (ablation): the setting that maximises the mean expected fractional variance reduction of all four parameters, estimated with a linear minimum-mean-square-error (LMMSE) estimator from one Poisson count under the current posterior.
5. **Target-aware design** (proposed): the same LMMSE criterion restricted to the defect thickness, the quantity that governs the device-relevant loss.

Methods 4 versus 5 isolate the effect of making the design aware of the target.

## Quantum baseline and amortised regressors

Amortised regressors map one 40-point uniform scan (log counts) directly to the defect thickness, trained on simulated labelled scans (dose 10^6 photons per point). Classical: ridge and a multilayer perceptron (MLP) on all 40 features; two MLPs on four principal components (8-4 and 32-32 hidden units). Quantum: a 4-qubit data re-uploading variational quantum circuit (VQC) with two layers (encode, rotate, entangle) and a linear Pauli-Z read-out, the model class analysed by Jerbi et al. (2023). It is simulated exactly with a batched PyTorch statevector routine, which a unit test checks against PennyLane. Ablation: one layer instead of two. Noise sweep: finite-shot read-out noise (1000 and 100 shots) at test time.

## Metrics

Root-mean-square error (RMSE) of the posterior-mean thickness (nm), 90 % credible-interval width and coverage, accuracy of the screening decision "thickness above 1.5 nm" (posterior probability above 0.5), and RMSE paired by seed against the uniform scan.

## Protocol

- Main comparison: 5 seeds (independent particle libraries), 6 true parameter draws per seed, 48 steps, all five methods.
- Robustness sweep: true background of 1, 10, 50 and 200 counts per step with the analyst assuming 1; uniform scan versus target-aware design; 3 seeds, 4 draws per seed.
- Amortised regressors: 3 seeds, 64, 256 and 1024 training scans, 400 test scans.
- Scaling in photons is read from the error trajectory (checkpoints every 4 steps).
- Everything is simulation on a central processing unit and a small GPU-free statevector; no hardware data.
