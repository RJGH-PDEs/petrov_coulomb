# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code implementing the **Galerkin--Petrov spectral method** for the Landau collision operator in the **Coulomb interaction case** (λ = −3). This is a companion to the Maxwell-molecules code in `rel_landau/` at `/Users/rjgh/Documents/Research/Projects/rel_landau`. The basis and test functions, mass matrix, and time evolution are the same as in `rel_landau`; only the quadrature strategy differs because the Coulomb kernel |u|^{λ} = |u|^{−3} is **singular at u = p−q = 0**.

The mathematical reference (read-only) is the classical Landau companion paper at `/Users/rjgh/Documents/Research/Latex/classical_landau/main.tex`. The Coulomb experiment is written up in `sec:num-coulomb`; the quadrature derivation is in `sec:singularity`.

## Git branches

- **`main`**: original code as first committed (less complete — `special_quad/` prototype, no time evolution, no plotting)
- **`development`**: current working branch — all production code, TACC results, time evolution, plots

Always work on `development`.

## Key algorithmic difference from rel_landau: change of variables

In `rel_landau`, the collision tensor integrates over (p, q) independently. For Coulomb this fails because |p−q|^{−3} is non-integrable without special treatment.

**Solution:** change variables (p, q) → (p, u) where u = p − q, so q = p − u. The Coulomb singularity is now in the radial variable |u|, which is handled by a **two-part special quadrature rule** in |u| (see `torsten_quad/`). The angular variable û = u/|u| uses an ordinary Lebedev rule.

The Gaussian weight for the q-trial function expands as:

    e^{−|q|²/2} = e^{−|p−u|²/2} = e^{−|p|²/2} · e^{p·u} · e^{−|u|²/2}

- `e^{−|p|²/2}` is absorbed by the p-radial GL weights.
- `e^{−|u|²/2}` is absorbed by the u-radial special weights (`torsten_quad/weight.pkl`).
- `e^{p·u}` remains **explicitly in the integrand** — see `parallel/integrand.py`, `inner_product`.

## Directory structure

```
petrov_coulomb/
├── torsten_quad/           special radial quadrature for |u| (the singular variable)
│   ├── quadrature.py       generates radius.pkl, weight.pkl, weight_exp.pkl (run once)
│   ├── concatenate.py      builds special_quad.pkl: two-piece rule (20-pt + 5-pt GL)
│   ├── integration.py      test: verifies the rule integrates r³ correctly
│   ├── to_integrate.py     defines h(r) = r³ (test integrand)
│   └── *.pkl               precomputed rule files (committed)
├── full_quad/              full 4D tensorized quadrature (p-radial × p-angular × u-radial × u-angular)
│   ├── serial_quadrature.py  builds and saves full_quad/quadrature.pkl (14700 pts total)
│   ├── unpack_quad.py      unpack [radial_p, ang_p, special_u, ang_u] → (weight, rp,tp,pp, ru,tu,pu)
│   └── quadrature.pkl      precomputed full quadrature (committed)
├── parallel/               MAIN PRODUCTION CODE — run from this directory
│   ├── parallel.py         outer loop over test fn (k,l,m); parallel inner loop over select; saves results/operator.pkl
│   ├── recompute.py        re-runs only previously-detected non-zeros (faster TACC re-check)
│   ├── landau.py           operator() and operator_parallel(): accumulate weight×integrand over quadrature
│   ├── integrand.py        integrand = weight × e^{p·u} × ψ_s(p) × ψ_r(q)
│   ├── derivatives.py      weight_new(k,l,m): symbolic gradient + hessian of test fn; weight_evaluator()
│   ├── weight_evaluator.py weight_evaluator_u(): converts (p,u) → (p,q) then calls weight_evaluator (with −1 sign)
│   ├── change_var.py       change_variables(rp,tp,pp,ru,tu,pu) → (rq,tq,pq) via Cartesian
│   ├── test_func.py        test(k,l,m,r,t,p): unweighted trial function φ_{k,l,m} (numpy); f_integrated = φ_s × φ_r
│   ├── to_numpy.py         converts sympy symbolic expressions → lambdified numpy callables (speedup)
│   └── unpack_quad.py      same as full_quad/unpack_quad.py
├── sparse_operator/
│   ├── compute_sparse.py   post-process: threshold non-zeros → simple indices → dense → CSR matrices
│   └── sparse_operator.pkl precomputed sparse operator (from operator_13913_nonegative.pkl, tol=0.03)
├── time_evolution/
│   ├── time_evolution.py   forward Euler: f_{n+1} = f_n + τ·M⁻¹·Q(f,f); τ=1e−6; saves coeff snapshots to plot/coeff/
│   ├── bilinear_operator.py  landau(so, f, result): result[i] = f^T · so[i] · f
│   ├── conservation_check.py  loads coeff snapshots, computes M·f, saves mass/momentum/energy to conservation/coulomb.csv
│   ├── conservation/       CSV output from conservation_check.py
│   └── mass_inverse.pkl    M⁻¹ (same mass matrix as rel_landau; precomputed)
├── plot/
│   ├── plot_overlay.py     overlay plot: multiple times on z-axis cut, plasma colormap, IC annotation; paper-ready
│   ├── plot_conservation.py  3-panel conservation plot (mass / momentum / energy proxy vs t); paper-ready
│   ├── plot.py             original single-snapshot plot (not paper-ready)
│   ├── lc.py               linear_comb(): reconstructs Σ α_j φ_j (unweighted; caller adds e^{−r²/2})
│   ├── test_func.py        same as parallel/test_func.py
│   ├── coeff/              coefficient pkl snapshots from time evolution (steps 1, 10, ..., 1000000)
│   └── figures/            output figures; paper figures are coulomb_relaxation_overlay.png and coulomb_conservation.png
├── operator/               EARLIER SERIAL PROTOTYPE — not production code
├── landau_weight/          EARLIER EXPLORATORY CODE — not production
├── gaussLegendre/          scratch test for GL quadrature
├── results/                operator pkl files from TACC runs (see TACC history below)
└── tacc/                   SLURM scripts and output logs
```

## The integrand (what is being computed)

The collision tensor entry `Q_i[s, r]` (test = i, trial-at-p = s, trial-at-q = r) is:

    Q_i[s,r] = ∫∫ weight(φ_i; p, u) · e^{p·u} · φ_s(p) · φ_r(q=p−u) dp du

where `weight(φ_i; p, u)` is (from `derivatives.py`, `weight_evaluator`):

    weight = [ −2u·(∇φ_i(p) − ∇φ_i(q)) + ½ tr(S(u)·(H_i(p) + H_i(q))) ] × C_{l_i}

with S(u) = |u|²I − u⊗u (Λ=1), H_i = Hessian of φ_i, and C_{l_i} = `spher_const(l_i, m_i)` (spherical harmonic normalization, multiplied separately — not part of the symbolic φ_i).

`weight_evaluator_u` calls `weight_evaluator` with a **−1 sign** to account for the (p,q)→(p,u) Jacobian.

## Index and data conventions

- **Basis function** identified by `(k, l, m)`: same as rel_landau.
- **Flat index**: `ind(k, l, m, L) = (L+1)²·k + l² + (m+l)` where `L = n−1`; for n=3, L=2 → same formula as rel_landau's `n²k + l² + (m+l)`. Key indices: mass=0, Py=1, Pz=2, Px=3, energy proxy=9.
- **`select`**: `[kp, lp, mp, kq, lq, mq]` — 2 basis function indices (trial at p, trial at q). Test function index is the outer loop in `parallel.py`.
- **Result entry**: `[[k,l,m], [kp,lp,mp,kq,lq,mq], value]`.
- **Quadrature unpack order**: `weight, rp, tp, pp, ru, tu, pu` (p-radial, p-angles, u-radial, u-angles).

## Special quadrature for |u| (`torsten_quad/`)

A **two-piece radial rule**:

1. **First piece** (20 nodes in `radius.pkl`, weights in `weight.pkl`): custom nodes for the inner region, likely Gauss-Jacobi tailored to the Coulomb weight.
2. **Second piece** (5-pt Gauss-Legendre on [0,1]): integrand transformed by `(1−x)/x` for the tail.

Concatenated into `torsten_quad/special_quad.pkl` by `concatenate.py`. Total: 25 radial nodes for |u|.

## Running things

Scripts use bare module imports and hard-coded relative paths — run each from its own directory:

```bash
cd torsten_quad    && python concatenate.py        # build special_quad.pkl (already committed)
cd full_quad       && python serial_quadrature.py  # build full_quad/quadrature.pkl (already committed)
cd parallel        && python parallel.py           # compute operator tensor (expensive; run on TACC)
cd sparse_operator && python compute_sparse.py     # post-process → sparse_operator.pkl
cd time_evolution  && python time_evolution.py     # forward Euler time integration → plot/coeff/
cd time_evolution  && python conservation_check.py # compute conservation CSV
cd plot            && python plot_overlay.py       # paper-ready overlay figure
cd plot            && python plot_conservation.py  # paper-ready conservation figure
```

Dependencies: `numpy`, `scipy`, `sympy`, `matplotlib`, `pylebedev`.

## TACC run history and current status

Multiple runs on Stampede2 (alloc DMS23021, job script `tacc/job.sh`). All runs used the same 14700-point quadrature from `full_quad/quadrature.pkl`. The numeric suffix in pkl filenames is the **SLURM job ID**, not the quadrature size.

| File | SLURM job | Notes |
|---|---|---|
| `operator_957.pkl` | 2106969 | Early run |
| `operator_1069.pkl` | 2110162 | — |
| `operator_12811.pkl` | 2121238 | — |
| `operator_13913.pkl` | 2122599 | Had a sign error (missing −1 Jacobian) |
| `operator_13913_nonegative.pkl` | 2130870 | Sign corrected; current best result |
| `operator_numpy.pkl` / `recomputed_numpy.pkl` | — | Numpy-accelerated versions |
| `operator_sympy.pkl` / `recomputed_sympy.pkl` | — | Sympy-only versions |

**Current best result**: `operator_13913_nonegative.pkl` → threshold tol=0.03 → `sparse_operator/sparse_operator.pkl`.

**Sign issue**: The −1 factor from the (p,q)→(p,u) Jacobian was initially missing in `weight_evaluator_u`. Fixed; the "nonegative" pkl reflects this correction.

**Time evolution**: Completed. Snapshots in `plot/coeff/` at steps 1, 10, 100, 1000, 2000, …, 1000000 (τ=1e−6; physical time t=0 to t=1). IC: α_{0,0,0}=1, α_{1,0,0}=−0.6 (isotropic double-hump).

**Conservation behavior**: Momentum conserved to machine precision. Mass and energy proxy drift by ~O(10⁻¹¹) relative over t=1 — because the Coulomb `parallel.py` does not skip conservation-law test functions (unlike `rel_landau`), so small quadrature errors in those rows accumulate.

## Paper figures

Produced by `plot/plot_overlay.py` and `plot/plot_conservation.py`; copied to `classical_landau/figs/coulomb/` and included in `sec:num-coulomb`.

- `coulomb_relaxation_overlay.png` — z-axis cut, 5 times, plasma colormap
- `coulomb_conservation.png` — 3-panel mass/momentum/energy proxy vs t

## Gotchas

- `test_func.py`'s `test()` is the **unweighted trial function φ_{k,l,m}** (no Gaussian weight), despite the name — same quirk as rel_landau.
- `weight_evaluator_u` multiplies by **−1** (Jacobian); `weight_evaluator` does not — don't confuse the two.
- `to_numpy.py` (in `parallel/` and `operator/`) converts sympy → lambdified numpy callables. Always use the numpy path for TACC runs; sympy is orders of magnitude slower.
- `spher_const(l,m)` is **not** baked into the symbolic φ in `derivatives.py` — multiplied separately at the end of `weight_evaluator`. Missing it doubles the angular normalization.
- `operator/` and `landau_weight/` are earlier prototypes; ignore them.
- `special_quad/` on `main` branch is a prototype; `torsten_quad/` on `development` is the refined version.
