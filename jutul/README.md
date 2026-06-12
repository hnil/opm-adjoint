# Jutul/JutulDarcy cross-validation helpers (T6)

A dedicated Julia environment plus helper code for comparing the
opm-adjoint module against JutulDarcy — forward simulations first
(gradient comparisons are only meaningful once the forward runs agree),
then adjoint sensitivities.

## One-time setup

```bash
julia --project=$(pwd) -e 'using Pkg; Pkg.add(["Jutul", "JutulDarcy"]); Pkg.precompile()'
```

This writes `Project.toml`/`Manifest.toml` here (commit them: the pinned
manifest makes results reproducible) and precompiles into the shared
depot. All scripts then start fast with `julia --project=<this dir>`.

## Usage

```julia
julia --project=<this dir>
include("OPMAdjointJutul.jl"); using .OPMAdjointJutul

case, result = run_forward("SPE1CASE1.DATA")
save_well_curves_csv("jutul_PROD.csv", result, "PROD")

# twin misfit objective (matchref-equivalent) + parameter sensitivities
ref_case, ref_result = run_forward("SPE1CASE1_REF.DATA")
obj  = well_rate_misfit_objective(ref_result, :PROD, :gas)
sens = parameter_sensitivities(case, result, obj)
```

## Scripts

- `../tests/run-jutul-forward-compare.sh` — stage 1: forward well-curve
  comparison OPM vs Jutul (unit-agnostic normalized shapes + unit-factor
  report). Run this before any gradient comparison.

Notes:
- Jutul and opm-adjoint share conventions: SI units internally, producer
  rates negative — the misfit objective above is term-for-term the same
  as `--adjoint-objective=matchref:...`.
- `parameter_sensitivities` uses JutulDarcy's parameter-AD adjoints; the
  porosity gradient compares against opm-adjoint via
  `dJ/dporo_i = g_pvmult_i / poro_i`.
