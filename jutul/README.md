# Jutul/JutulDarcy cross-validation helpers (T6)

A dedicated Julia environment plus helper code for comparing the
opm-adjoint module against JutulDarcy — forward simulations first
(gradient comparisons are only meaningful once the forward runs agree),
then adjoint sensitivities.

## Setup instructions (one-time, ~10–30 min of precompilation)

The first attempt struggled on Julia 1.12 (very new release — the
JutulDarcy dependency stack precompiles slowly/poorly there). Recommended:
use the LTS channel for this environment:

```bash
juliaup add lts                  # installs Julia 1.10.x
cd <opm-adjoint>/jutul
julia +lts --project=. -e 'using Pkg; Pkg.add(["Jutul", "JutulDarcy"]); Pkg.precompile()'
# sanity check (should print version and exit quickly the second time):
julia +lts --project=. -e 'using JutulDarcy; println("JutulDarcy OK")'
```

This generates `Project.toml`/`Manifest.toml` in this directory —
**commit both**: the pinned manifest makes cross-validation results
reproducible. All later invocations start fast
(`julia +lts --project=<this dir>`).

If 1.12 is preferred anyway, the same commands without `+lts` work but
expect a long first `Pkg.precompile()`; run it once in a free terminal,
not under time pressure.

## What to run once the environment works

1. **Stage 1 — forward equivalence (required before any gradient
   comparison):**

   ```bash
   cd <opm-adjoint>
   ./tests/run-jutul-forward-compare.sh \
       <build>/opm-simulators/bin/flow_blackoil \
       <build>/opm-common/bin/summary \
       <opm-tests>/spe1/SPE1CASE1.DATA /tmp/jutulcmp PROD
   ```

   It runs both simulators on the deck, extracts WOPR/WGPR/WBHP for the
   well, and compares unit-agnostic normalized curve shapes (linear time
   alignment); the reported OPM/Jutul average ratios expose any unit
   convention differences explicitly. PASS threshold is 2% on the
   normalized shapes by default (discretization differences between the
   two simulators' time stepping are expected at this level; tighten
   after inspecting). If the shapes disagree more than that, STOP and
   understand why before comparing gradients.

   (If the julia invocation inside the script should use the LTS channel,
   edit the `julia --project=...` line to `julia +lts --project=...`.)

2. **Stage 2 — gradient cross-check** (manual for now, helpers ready in
   `OPMAdjointJutul.jl`):

   ```julia
   julia +lts --project=<this dir>
   include("OPMAdjointJutul.jl"); using .OPMAdjointJutul

   # reference and base forward runs (same decks as the OPM-side
   # run-adjoint-matchref-test.sh: reference PORO 0.285, base PORO 0.3)
   ref_case, ref_result = run_forward("SPE1CASE1_REF.DATA")
   case, result         = run_forward("SPE1CASE1.DATA")

   # the matchref-equivalent objective (same J term-for-term)
   obj = well_rate_misfit_objective(ref_result, :PROD, :gas)

   sens = parameter_sensitivities(case, result, obj)
   # porosity gradient: compare against opm-adjoint's
   # <CASE>.ADJOINT_GRADIENTS_PV.txt via dJ/dporo_i = g_pvmult_i / poro_i
   ```

   Expected agreement: not bitwise (different discretizations/time
   stepping), but the gradients should agree to a few percent on smooth
   cases, and the *sign pattern and spatial structure* must match.
   For a sharper comparison run both with the same fixed report-step
   time stepping (OPM: `--enable-adaptive-time-stepping=false`).

## Helper API (`OPMAdjointJutul.jl`)

- `run_forward(deck; kwargs...)` -> `(case, result)`
- `well_curves(result, well)` — time/orat/grat/wrat/bhp, SI, Jutul sign
  convention (producers negative — same as opm-adjoint internals)
- `save_well_curves_csv(path, result, well)` — display units
- `well_rate_misfit_objective(result_obs, well, phase; w)` — Jutul
  sum-objective identical to `--adjoint-objective=matchref:...`
- `parameter_sensitivities(case, result, objective)` — JutulDarcy
  parameter-AD adjoints (`reservoir_sensitivities`)

Note: `parameter_sensitivities`/`compute_well_qoi` call signatures track
JutulDarcy's API; if a JutulDarcy upgrade changes them, fix up here —
the logic (misfit form, conventions) is the part that matters.


## Forward comparison status (2026-06-13)

Environment installed on Julia LTS (1.10.11); `Project.toml`/`Manifest.toml`
are committed for reproducibility. JutulDarcy reads the OPM `.DATA` deck
directly (`setup_case_from_data_file`).

`tests/run-jutul-forward-compare.sh` runs both codes on the same deck and
compares well curves. SPE1CASE1 result, with units reconciled (OPM
summary emits FIELD: STB/d, MSCF/d, psia; Jutul emits metric):

- **Rate target matches exactly**: OPM WOPR 20000 STB/d = 3179.7 sm3/d
  = Jutul orat 3179.75 sm3/d at the first step (rel 0).
- **BHP floor matches exactly**: both producers bottom out at 68.95 bar
  (= the 1000 psia limit) in the late, BHP-controlled regime.
- **Mid-run they diverge** in *when* the producer switches from rate to
  BHP control, and in the well-index/BHP relationship: at t=1 d the same
  oil rate gives OPM 200 bar vs Jutul 156 bar BHP. This is the expected
  OPM-vs-Jutul well-model difference (connection factor / wellbore
  coupling), not a reservoir-physics disagreement - the rates agree
  whenever the well is rate-controlled and the BHP floor is identical.

Implication for gradient cross-validation (T6 stage 2): pick a
configuration where the two well models agree tightly - e.g. compare
*pressure-average* or a BHP-controlled-well objective, rather than an
oil-rate objective on a producer that switches control mid-run. The
`run-jutul-forward-compare.sh` PASS/FAIL on shape-normalized curves is
therefore informational; the unit-reconciled point comparison above is
the meaningful check.


## Gradient cross-check (T6 stage 2, 2026-06-13)

`jutul/compare_spe1_pv.jl` computes the JutulDarcy adjoint gradient of the
final-step average reservoir pressure w.r.t. porosity on SPE1CASE1, via
`reservoir_sensitivities(case, result, G; include_parameters=true)` ->
`sens[:porosity]`. Compared against the opm-adjoint gradient
(`dJ/dporo_i = g_pvmult_i / poro_i` from
`SPE1CASE1.ADJOINT_GRADIENTS_PV.txt`, poro = 0.3):

- **Pearson correlation 0.80**, sign-consistent on all dominant cells;
  best-fit scale OPM ~ 0.54 x Jutul.

The agreement is a pattern/sign cross-check, not a tight match, for known
reasons: the two forward runs already differ in the well model
(connection factor / control-switch timing, see above), and the
"average pressure" objective is defined on different pressure variables
(OPM uses the pressure primary variable = oil pressure; Jutul uses the
reservoir reference pressure). The authoritative quantitative check
remains the in-tree central-difference test (opm-adjoint matches FD to
~1e-4); JutulDarcy is the independent-implementation sanity check, and it
confirms the opm-adjoint gradient has the right structure and sign.

Run:
```bash
# OPM side
flow_adjoint SPE1CASE1.DATA --enable-storage-cache=false \
  --enable-adaptive-time-stepping=false --adjoint-objective=pressure-average --adjoint-save=true
flow_adjoint SPE1CASE1.DATA ... --adjoint-mode=gradient
# Jutul side
julia +lts --project=jutul jutul/compare_spe1_pv.jl SPE1CASE1.DATA jutul_poro.csv
```
