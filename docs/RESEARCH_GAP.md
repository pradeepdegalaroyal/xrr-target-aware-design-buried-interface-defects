# Research gap

## Context

Defects at buried interfaces (oxide, sub-oxide and contamination layers between a superconducting film and its substrate) are a suspected source of two-level-system (TLS) loss in superconducting quantum devices. Synchrotron X-ray reflectivity (XRR) is a non-destructive probe of such buried layers, but beam time is scarce and each measurement setting costs photons. A theory-based machine learning (ML) platform therefore needs two parts: a physical forward model that links defect parameters to the measured curve, and a rule that decides which setting to measure next.

## State of the art in the supplied literature

The brief contains no paper on X-ray characterisation or on experiment design, so the table below places the nearest work by its relation to this problem. The brief labels every record with the generic role "other"; the "topical role" column is therefore assigned by the author of this repository and is not taken from the brief.

| Paper (Author et al., Year, DOI) | Topical role | What it does | What it does NOT cover |
|---|---|---|---|
| Burkard et al., 2023, 10.1103/RevModPhys.95.025003 | Review (device physics) | Reviews semiconductor spin qubits, including the impact of noise through the filter-function formalism | No structural probe of the interface that produces the noise, and no measurement-selection rule |
| Sivak et al., 2023, 10.1038/s41586-023-05782-6 | Landmark (device) | Error-corrected logical qubit beyond break-even (coherence gain 2.27 ± 0.07) using model-free reinforcement learning | Learning is applied to control, not to characterising the materials that limit coherence |
| Evered et al., 2023, 10.1038/s41586-023-06481-y | Landmark (device) | Two-qubit gates with 99.5 % fidelity on up to 60 atoms in parallel; characterises physical error sources | Neutral atoms, so no buried solid-state interfaces |
| Ma et al., 2023, 10.1038/s41586-023-06438-1 | Landmark (device) | Neutral-atom qubit with mid-circuit erasure conversion and structured error models | Error structure is engineered, not diagnosed from materials data |
| Storz et al., 2023, 10.1038/s41586-023-05885-0 | Landmark (device) | Loophole-free Bell test with superconducting circuits over a 30 m cryogenic link | Demonstrates the platform, not its materials limits |
| Jerbi et al., 2023, 10.1038/s41467-023-36159-y | Landmark (quantum ML baseline) | Shows that data re-uploading circuits map to linear quantum models and analyses their resource needs | No scientific-instrument data and no sequential decision task |
| Suaud et al., 2026, 10.1016/j.ccr.2026.218440 | Review (theory) | First-principles extraction of effective spin Hamiltonians and anisotropic interactions | Not linked to measured data or to the choice of the next measurement |
| Wang et al., 2027, 10.15302/frontphys.2027.025202 | Recent (theory) | First-principles magnon-phonon coupling in a CrI3 monolayer | Static calculation, no inference from experiment |
| Gorgon et al., 2023, 10.1038/s41586-023-06222-1 | Adjacent (materials) | Molecular spin-optical interface for quantum information science | Material design, not defect characterisation |
| Dutt et al., 2024, 10.1038/s41578-024-00668-z | Review (materials) | Integrated nonlinear and quantum photonic materials | Photonic materials only |
| Fisher et al., 2023, 10.1146/annurev-conmatphys-031720-030658 | Review (adjacent) | Monitored random circuits, measurement-induced transitions | Measurement as a physical process, not as a design variable |
| Xiao and Pedrycz, 2023, 10.1109/TPAMI.2022.3167045 | Adjacent (uncertainty) | Notes that expressing uncertainty in knowledge remains open in artificial intelligence (AI) systems | Fuzzy and quantum-like fusion, no physical forward model |

## Gap

Recent device papers push fidelities to the surface-code range and beyond, and the review literature points at noise from materials as the next limit. The theory papers supply first-principles forward models. What is missing in the supplied record is the link between them: a quantitative, photon-budgeted rule for choosing the next structural measurement so that the defect parameter that matters for the device is pinned down with the fewest photons, together with an honest test of whether learned (including quantum) surrogates can stand in for the physics when labelled simulations are scarce.

## Testable hypotheses (simulation scale)

All experiments are simulations of a five-layer Nb-oxide / Nb / defect / sapphire stack with generic, illustrative optical constants. Nothing here is fitted to beamline data.

- **H1.** At an equal photon budget (48 measurement steps of 3 x 10^4 incident photons), a target-aware sequential design lowers the root-mean-square error (RMSE) of the defect-layer thickness by at least 20 % relative to a uniform scan.
- **H2.** That advantage persists when the detector background is up to 200 times the value assumed by the analyst (RMSE ratio to the uniform scan stays below 1).
- **H3.** In the label-scarce regime (64 to 1024 simulated, labelled scans) a 4-qubit data re-uploading circuit reaches a lower test RMSE than classical regressors on the same inputs.

Success metric: RMSE (nm) of the posterior-mean defect thickness, paired by seed against the uniform scan; secondary metrics are 90 % credible-interval coverage and accuracy of a screening decision at 1.5 nm.

## Novelty

This study measures, in one reproducible simulation, how much a target-aware sequential design (and several standard alternatives) changes the photon cost of quantifying a buried-interface defect layer, and tests a quantum surrogate on the same task, which none of the cited papers do.
