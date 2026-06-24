# OPM adjoint — project status & quickstart

_Last updated: 2026-06-24_

Adjoint-based gradients for the OPM flow simulator: gradients of well-curve-matching
(and other) objectives with respect to porosity/PV, transmissibility, permeability,
end-point scaling, and well controls — computed by the adjoint method.

---

## 1. Status summary

**Milestones A, B and C are complete and finite-difference (FD) verified.** The core
deliverable — adjoint gradients of well-matching objectives w.r.t.
poro/perm/trans/end-points/well-controls — works on cases ranging from 1-cell debug
decks up to field models, and is guarded by a green ctest suite (**15/15**).

### Development layout

- **`opm-adjoint`** — separate downstream module (the `opm-flowgeomechanics` pattern),
  built against a **near-stock `opm-simulators`**. Recording hooks live in a derived
  `AdjointFlowProblem` (virtual `beginEpisode`/`endTimeStep`); the backward replay/solver
  driver uses only public simulator APIs. Branch `master`, HEAD `b231773`.
- **`opm-simulators`** — branch `adjoint-hooks` (master + a handful of small accessor
  commits: `B/C/D` getters on `StandardWellEquations`, a `SummaryState` assembly seam,
  half-trans exposure on `Transmissibility`, `assembleWellEqGivenControls`). HEAD
  `9b07194fe`.
- **`opm-tests`** — branch `adjoint` (adds `adjoint_tests/MODEL_*_DEBUG` and
  `SPE1_GRPCTRL` decks).

### What works

| Area | Status |
|---|---|
| **Milestone A — exact backward replay** | SPE1 123/123, SPE9 21/21, MODEL_1D_DEBUG 7/7, SPE1_GRPCTRL 120/120 **bitwise**. Full **Norne** (44 431 cells, 36 wells, group ctrl + shut wells, hysteresis, VAPOIL, faults) replays **end-to-end**, ~90 % bitwise (remainder at well-event / hysteresis boundaries — documented v1 approximation). |
| **Milestone B — adjoint solve** | Backward Lagrange sweep, Schur-reduced, UMFPACK on the transposed system; iterative solves (ilu0 / cpr / **cprt**); **MPI** parallel record/replay + transposed solve (cross-rank transpose bug found & fixed). |
| **Milestone C — parameter gradients (all FD-verified)** | PV / pvmult, transmissibility / tmult, **perm** (half-trans chain rule), **well-control dJ/du** (the original 2019 goal), **end-point scaling**. |
| **Objectives** | pressure-average, BHP, well-curve matching (`matchref` / `matchsum` via `ESmry`), multi-term. |
| **Tests** | ctest **15/15**; FD-verified golden-reference regression set in `tests/references/`. |
| **JutulDarcy cross-check** | Set up (`jutul/`); pattern/sign-level agreement (forwards differ by well model). FD remains the authoritative oracle. |

### Key design decisions

- v1 forces `--enable-storage-cache=false` so the residual is a pure function of
  `(x_k, x_{k-1})` — replay needs a single linearization per substep.
- Archive = `serializeOp` buffers in an HDF5 file (`<CASE>.ADJOINT.h5`, parallel HDF5
  active) or a dependency-free directory store (`--adjoint-file=<dir>` without `.h5`).
- Gradient accuracy tracks the linear-solve reduction roughly linearly; default
  reduction `1e-14`.

### Known gaps / queue (low priority, none blocking)

- Report-step-0 startup bitwise for model2-like decks; Norne hysteresis / well-event
  boundary bitwise.
- **WGRUPCON / VREP** voidage-replacement replay-inexactness — fix _designed_ (Stage-2:
  carry the converged effective control target per well in the `WellAssemblyState`
  bundle so `assembleControlEqInj/Prod` read it directly), **not yet implemented**.
- `FlexibleCellEvaluator` (C.2) for exact parameter-AD; end-point scaling is currently
  via cell-local parameter FD.
- T6 JutulDarcy gradient cross-check is pattern/sign-only.

The end-of-run `AdjointDeckWarnings` scan reports, per deck, every quantity the adjoint
does not handle exactly (approx-gradient: hysteresis, DRSDT/DRVDT, VAPPARS, guide-rate
group allocation, well explicit quantities; approx-replay: voidage group control;
neglected: aquifers, networks). SPE1 prints nothing (clean).

---

## 2. Checkout

The working tree is a set of side-by-side OPM module repos. The adjoint work lives on
three non-default branches:

```bash
# opm-simulators: small accessor hooks on top of master
git -C opm-simulators checkout adjoint-hooks

# opm-tests: adjoint debug decks + SPE1_GRPCTRL
git -C opm-tests checkout adjoint

# opm-adjoint: the module itself (default branch)
git -C opm-adjoint checkout master
```

`opm-common` and `opm-grid` stay on their normal branches. `opm-adjoint/dune.module`
declares the dependency chain: `dune-istl opm-common opm-grid opm-simulators`.

---

## 3. Compile

The module builds like any Dune/OPM module. The existing release tree is
`builds/release` (Ninja, `CMAKE_BUILD_TYPE=Release`, parallel HDF5 + UMFPACK).

```bash
cd /Users/hnil/Documents/OPM/opm_clean/builds/release

# (re)configure — only needed for a fresh tree
cmake -G Ninja \
  -C ../cmake_minimal.cmake \
  -DCMAKE_PREFIX_PATH=/Users/hnil/Documents/GITHUB/OPM/dune_installed_all \
  -DCMAKE_BUILD_TYPE=Release \
  /path/to/opm-adjoint            # plus the other modules as usual

# build just the adjoint binary
cmake --build . --target flow_adjoint -j8
```

Producing `builds/release/opm-adjoint/flow_adjoint`.

> **Build gotcha (important):** downstream executables must use
> `use_additional_optimization(TARGET ...)`. OPM builds Release with `-UNDEBUG`
> (`WITH_NDEBUG=OFF`); an `NDEBUG` mismatch against the library is an ABI break that
> crashes inside the material law during equilibration. This is already handled in the
> module's CMake.

---

## 4. Run an example

`flow_adjoint` is a single binary. Mode is selected by `--adjoint-mode`
(default = plain forward; `replay`, `gradient`, `objective`). The forward stage
**records** with `--adjoint-save=true`; later stages read that archive.

Set a convenience variable:

```bash
FLOW=/Users/hnil/Documents/OPM/opm_clean/builds/release/opm-adjoint/flow_adjoint
DECK=/Users/hnil/Documents/OPM/opm_clean/opm-tests/spe1/SPE1CASE1.DATA
```

### 4a. Exact backward replay (Milestone A — verifies bitwise re-linearization)

```bash
mkdir -p /tmp/spe1_adj

# forward run, recording each accepted substep's converged residual/Jacobian too
$FLOW $DECK --output-dir=/tmp/spe1_adj \
    --threads-per-process=1 --enable-storage-cache=false \
    --adjoint-save=true --adjoint-save-system=true

# backward replay: re-create every substep's system and compare to the stored one
$FLOW $DECK --output-dir=/tmp/spe1_adj \
    --threads-per-process=1 --enable-storage-cache=false \
    --adjoint-mode=replay --adjoint-replay-tolerance=1e-12
```

The replay log ends with `Adjoint replay finished`; SPE1 matches 123/123 substeps
bitwise. The wrapper `tests/run-adjoint-replay-test.sh <flow> <deck> <out-dir> [tol]`
does exactly this.

### 4b. Compute a gradient (Milestones B+C)

Uses a group-control field case with per-cell PORO. **Gradient runs disable adaptive
time-stepping** so a parameter perturbation cannot change the substep sequence (which
would swamp any FD comparison):

```bash
DECK=/Users/hnil/Documents/OPM/opm_clean/opm-tests/adjoint_tests/SPE1_GRPCTRL/SPE1_GRPCTRL.DATA
OUT=/tmp/spe1grp
mkdir -p $OUT

STRICT="--enable-storage-cache=false --enable-adaptive-time-stepping=false \
        --threads-per-process=1 --tolerance-cnv=1e-9 --tolerance-mb=1e-11 \
        --adjoint-objective=pressure-average"

# 1) forward + record
$FLOW $DECK --output-dir=$OUT $STRICT --adjoint-file=$OUT/a --adjoint-save=true

# 2) backward sweep -> dJ/d(porosity multiplier) per cell
$FLOW $DECK --output-dir=$OUT $STRICT --adjoint-file=$OUT/a --adjoint-mode=gradient
#   => $OUT/<CASE>.ADJOINT_GRADIENTS_PV.txt

# 3) (optional) just the objective value, e.g. for an FD check
$FLOW $DECK --output-dir=$OUT $STRICT --adjoint-file=$OUT/a --adjoint-mode=objective
#   => log line "Adjoint objective J = ..."
```

Objectives: `--adjoint-objective=pressure-average | bhp:<WELL> | matchref:<ref>:<W>.<p>
| matchsum:<ref>:<W>.<p>+...`. Parameter families are written to
`ADJOINT_GRADIENTS_{PV,TRANS,PERM,...}.txt`; well-control multipliers to
`ADJOINT_LAMBDA_WELLS.txt`.

> **Reference-summary caveat:** gradient/objective/replay runs truncate summary files in
> their output dir at startup, so a `matchref` reference summary must live **outside** the
> run's `--output-dir`.

### 4c. Run the test suite

```bash
cd /Users/hnil/Documents/OPM/opm_clean/builds/release/opm-adjoint
ctest                      # 15/15, ~30 s
ctest -R adjoint_fd        # just the FD-vs-adjoint gradient checks
```

Individual `tests/run-adjoint-*.sh` scripts (replay, FD per parameter family, matchref,
MPI, storage-cache equivalence) take `<flow_adjoint> <deck> <out-dir>` and can be run
standalone. They are `bash` scripts — on this machine run them with `bash`, not `zsh`
(the test one-liners rely on word-splitting).

---

## 5. Where to read more

- `adjoint_plan.md` — full design rationale and PR sequencing.
- `adjoint_status.md` — detailed feature catalog (incl. §6 explicitly-updated-quantities
  table that the end-of-run warning mirrors).
- `adjoint_refactoring.md` — the Stage-1/2/3 `WellAssemblyState` refactor that closes the
  remaining group-control / voidage replay gaps.
- `adjoint_testing.md` — per-test-level instructions.
- `jutul/README.md` — JutulDarcy cross-check setup (use the Julia LTS channel: `+lts`).
