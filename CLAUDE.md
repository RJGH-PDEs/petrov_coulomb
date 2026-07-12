# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code implementing the **Galerkin--Petrov spectral method** for the Landau collision operator in the **Coulomb interaction case** (λ = −3). This is a companion to the Maxwell-molecules code in `rel_landau/`. The basis and test functions, mass matrix, and time evolution are the same; only the quadrature strategy differs because the Coulomb kernel |u|^{λ} = |u|^{−3} is **singular at u = p−q = 0**.

The mathematical reference (read-only) lives in the write-up repo at `/Users/rjgh/Documents/Research/Latex/Relativistic Landau/` (core file `main.tex`).

## Key algorithmic difference from rel_landau: change of variables

In `rel_landau`, the collision tensor integrates over (p, q) independently. For Coulomb this fails because |p−q|^{−3} is non-integrable without special treatment.

**Solution:** change variables (p, q) → (p, u) where u = p − q, so q = p − u. The Coulomb singularity is now in the radial variable |u|, which is handled by a **two-part special quadrature rule** in |u| (see `torsten_quad/`). The angular variable û = u/|u| uses an ordinary Lebedev rule.

The Gaussian weight for the q-trial function expands as:

    e^{−|q|²/2} = e^{−|p−u|²/2} = e^{−|p|²/2} · e^{p·u} · e^{−|u|²/2}

- `e^{−|p|²/2}` is absorbed by the p-radial GL weights.
- `e^{−|u|²/2}` is absorbed by the u-radial special weights (`weight.pkl` = without exponential).
- `e^{p·u}` remains **explicitly in the integrand** — see `parallel/integrand.py`, `inner_product`.

## Directory structure

```
petrov_coulomb/
├── torsten_quad/           special radial quadrature for |u| (the singular variable)
│   ├── quadrature.py       generates radius.pkl, weight.pkl, weight_exp.pkl (run once)
│   ├── concatenate.py      builds special_quad.pkl: two-piece rule (20-pt + 5-pt GL)
│   ├── integration.py      test: verifies the rule integrates r³ correctly
│   ├── to_integrate.py     defines h(r) = r³ (test integrand)
│   └── *.pkl               precomputed rule files
├── full_quad/              full 4D tensorized quadrature (p-radial × p-angular × u-radial × u-angular)
│   ├── serial_quadrature.py  builds and saves full_quad/quadrature.pkl (14700 pts: 3×7×25×28?)
│   ├── unpack_quad.py      unpack [radial_p, ang_p, special_u, ang_u] → (weight, rp,tp,pp, ru,tu,pu)
│   └── quadrature.pkl      precomputed full quadrature (committed)
├── parallel/               MAIN PRODUCTION CODE — run from this directory
│   ├── parallel.py         outer loop over test fn (k,l,m); parallel inner loop over select; saves results/operator.pkl
│   ├── recompute.py        re-runs only previously-detected non-zeros (faster TACC re-check)
│   ├── landau.py           operator() and operator_parallel(): accumulate weight×integrand over quadrature
│   ├── integrand.py        integrand = weight × e^{p·u} × ψ_s(p) × ψ_r(q)
│   ├── derivatives.py      weight_new(k,l,m): symbolic gradient + hessian of test fn; weight_evaluator()
│   ├── weight_evaluator.py weight_evaluator_u(): converts (p,u) → (p,q) then calls weight_evaluator
│   ├── change_var.py       change_variables(rp,tp,pp,ru,tu,pu) → (rq,tq,pq) via Cartesian
│   ├── test_func.py        test(k,l,m,r,t,p): unweighted trial function φ_{k,l,m} (numpy); f_integrated = φ_s × φ_r
│   ├── to_numpy.py         converts sympy symbolic expressions → lambdified numpy callables (speedup)
│   └── unpack_quad.py      same as full_quad/unpack_quad.py
├── sparse_operator/
│   ├── compute_sparse.py   post-process: threshold non-zeros → simple indices → dense → CSR matrices
│   └── sparse_operator.pkl precomputed sparse operator (from operator_13913_nonegative.pkl, tol=0.03)
├── time_evolution/
│   ├── time_evolution.py   forward Euler: f_{n+1} = f_n + τ·M⁻¹·Q(f,f); τ=1e−6; saves coeff snapshots
│   ├── bilinear_operator.py  landau(so, f, result): result[i] = f^T · so[i] · f
│   └── mass_inverse.pkl    M⁻¹ (same mass matrix as rel_landau; precomputed)
├── plot/
│   ├── plot.py             1D reconstruction along x-axis; plots f(r)·e^{−r²/2}
│   ├── lc.py               linear_comb(): reconstructs Σ α_j φ_j (unweighted; plot.py adds the exp)
│   ├── test_func.py        same as parallel/test_func.py
│   ├── coeff/              coefficient pkl snapshots (1, 10, ..., 1000000 iterations)
│   └── figures/            corresponding snapshot plots
├── operator/               EARLIER SERIAL PROTOTYPE (not the current production code)
├── landau_weight/          EARLIER EXPLORATORY CODE (not production)
├── gaussLegendre/          scratch test for GL quadrature
├── results/                operator pkl files from TACC runs (see TACC history below)
└── tacc/                   SLURM scripts and output logs
```

## The integrand (what is being computed)

The collision tensor entry `Q_i[s, r]` (test = i, trial-at-p = s, trial-at-q = r) is:

    Q_i[s,r] = ∫∫ weight(φ_i; p, u) · e^{p·u} · φ_s(p) · φ_r(q=p−u) dp du

where `weight(φ_i; p, u)` is (from `derivatives.py`, `weight_evaluator`):

    weight = [ −2u·(∇φ_i(p) − ∇φ_i(q)) + ½ tr(S(u)·(H_i(p) + H_i(q))) ] × C_{l_i}

with S(u) = |u|²I − u⊗u (Coulomb kernel, Λ=1), H_i = Hessian of φ_i, and C_{l_i} = spher_const(l_i, m_i) (spherical harmonic normalization constant not included in the symbolic φ_i expression, multiplied separately).

`weight_evaluator_u` calls `weight_evaluator` with a **−1 sign** to account for the (p,q)→(p,u) Jacobian.

## Index and data conventions

- **Basis function** identified by `(k, l, m)`: same as rel_landau.
- **Flat index**: `ind(k, l, m, L) = (L+1)²·k + l² + (m+l)` where `L = n−1` (max l value); for n=3, L=2, same formula as rel_landau's `n²k + l² + (m+l)`.
- **`select`** in the Coulomb code: `[kp, lp, mp, kq, lq, mq]` — 2 basis function indices (trial at p, trial at q). The test function is the outer loop in `parallel.py`.
- **Result entry**: `[[k,l,m], [kp,lp,mp,kq,lq,mq], value]` — test function index, select pair, scalar value.
- **Quadrature unpack order**: `weight, rp, tp, pp, ru, tu, pu` (p-radial, p-angles, u-radial, u-angles).

## Special quadrature for |u| (`torsten_quad/`)

A **two-piece radial rule** that integrates h(r) = r^3 · (kernel singularity absorbed):

1. **First piece** (20 nodes in `radius.pkl`): custom nodes for the near-zero region; weights in `weight.pkl` (without e^{−r}) are used when the Gaussian is absorbed into the tensorized rule. These look like Gauss-Jacobi nodes tailored for the Coulomb singularity.
2. **Second piece** (5-pt Gauss-Legendre on [0,1]): integrand transformed by `(1−x)/x` to handle the complementary tail.

The two pieces are concatenated into `torsten_quad/special_quad.pkl` by `concatenate.py`.

## Running things

Scripts use bare module imports and hard-coded relative paths — run each from its own directory:

```bash
cd torsten_quad   && python concatenate.py     # build special_quad.pkl (already committed)
cd full_quad      && python serial_quadrature.py  # build full_quad/quadrature.pkl (already committed)
cd parallel       && python parallel.py            # compute operator tensor (expensive; run on TACC)
cd sparse_operator && python compute_sparse.py    # post-process → sparse_operator.pkl
cd time_evolution && python time_evolution.py      # forward Euler time integration
cd plot           && python plot.py                # reconstruction plot
```

Dependencies: `numpy`, `scipy`, `sympy`, `matplotlib`, `pylebedev`.

## TACC run history and current status

Multiple TACC runs on Stampede2 (alloc DMS23021, job script `tacc/job.sh`). The quadrature size in the filename suffix is the number of integration points:

| File | Points | Notes |
|---|---|---|
| `operator_957.pkl` | 957 | Early run |
| `operator_1069.pkl` | 1069 | — |
| `operator_12811.pkl` | 12811 | — |
| `operator_13913.pkl` | 13913 | Had a sign error |
| `operator_13913_nonegative.pkl` | 13913 | Sign corrected ("nonegative" suffix) |
| `operator_numpy.pkl` | — | Numpy-accelerated version |
| `operator_sympy.pkl` | — | Sympy-only version |
| `recomputed_numpy.pkl` | — | Recomputed non-zeros with numpy |
| `recomputed_sympy.pkl` | — | Recomputed non-zeros with sympy |

**Current best result**: `operator_13913_nonegative.pkl` — sign corrected, threshold tol=0.03, sparse operator built and saved to `sparse_operator/sparse_operator.pkl`.

**Sign issue history**: A −1 factor from the (p,q)→(p,u) Jacobian was initially missing. Corrected in `weight_evaluator_u` (multiply by −1). The "nonegative" pkl reflects this fix; see comment in `compute_sparse.py`.

**Time evolution**: Has been run (coeff/ contains snapshots 1–1000000). IC: `f[0]=1, f[9]=−0.6` (same double-hump as rel_landau). τ=1e−6.

## Gotchas

- `test_func.py`'s `test()` function is the **unweighted trial function φ_{k,l,m}** (no Gaussian), despite the name — same naming quirk as rel_landau.
- `weight_evaluator_u` multiplies by **−1** (the (p,q)→(p,u) Jacobian); `weight_evaluator` does not — don't confuse the two.
- `to_numpy.py` (in `parallel/` and `operator/`) converts sympy symbolic derivatives to lambdified numpy callables. Use the numpy path for any new TACC run; sympy is too slow at scale.
- `spher_const(l,m)` is **not** included in the symbolic test function `f` in `derivatives.py` — it is multiplied on separately at the end of `weight_evaluator`. Forgetting this doubles the angular normalization.
- The `operator/` and `landau_weight/` directories are earlier prototypes; the current production code is in `parallel/`.
- `special_quad/` on the `main` branch was a prototype; the `torsten_quad/` directory on `development` is the refined version (same idea, better rule).
