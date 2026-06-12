# Adjoint implementation — status and testing guide

*Companion to [adjoint_plan.md](adjoint_plan.md). Updated 2026-06-11.*

## Where development happens now: the `opm-adjoint` module

Development moved to a **separate module repo `opm_clean/opm-adjoint`**
(geomech-style) that builds against **unmodified opm-simulators master** —
no rebases, no CMakeLists conflicts. opm-simulators is checked out on
`master`; the in-tree `adjoint` branch is frozen at tag
`adjoint-milestone-A` as history (all code was moved to the module).

Module layout: `opm/adjoint/` (storage, recorder, replay, linalg —
moved unchanged), `AdjointFlowProblem` (recording via virtual
`beginEpisode`/`endTimeStep` overrides), `AdjointWellModel.hpp` (WGState
normalization via the protected-member-pointer pattern),
`examples/flow_adjoint.cpp` (single binary: forward+record by default,
`--adjoint-mode=replay` for the backward sweep). The superbuild includes
it automatically (top-level CMakeLists); binary:
`<build>/opm-adjoint/flow_adjoint`, tests in `<build>/opm-adjoint/bin/`.

Hard-won build lesson: downstream binaries must use
`use_additional_optimization(TARGET ...)` (or opm_add_executable) — the
opm convention `WITH_NDEBUG=OFF` compiles Release with `-UNDEBUG`, and an
NDEBUG mismatch against the opm-simulators library is an ABI break that
crashes in inlined material-law code.

Verified (Release + Debug, HDF5 backend): unit tests pass; SPE1CASE1
123/123 and MODEL_1D_DEBUG 7/7 substeps replay **bitwise** against
pristine master.

### Milestone B core: adjoint sweep + FD-verified gradients

`flow_adjoint --adjoint-mode=gradient` runs the backward adjoint sweep on
a recorded archive: replay each substep (bitwise verification kept on),
capture `Bdiag_k = -(V/dt) dStorage/dx|x_{k-1}` during the iteration-0
stage, form the Schur-reduced matrix via the public
`addWellContributions`, solve the transposed systems with UMFPACK, and
accumulate dJ/d(pvmult_i) and dJ/d(tmult_f) (storage/flux linearities;
values re-evaluated through the public templated TPFA kernels).
First objective: J = average pressure primary variable at the final
substep. Outputs `<CASE>.ADJOINT_GRADIENTS_{PV,TRANS}.txt`;
`--adjoint-mode=objective` evaluates J on a recorded archive (used by
the FD loop).

**T5 finite-difference verification — PASS** (`tests/run-adjoint-fd-test.sh`,
MODEL_1D_DEBUG, fixed report-step stepping, strict tolerances):

| cell | FD dJ/dporo | adjoint | rel. err |
|---|---|---|---|
| 0 | 3.783065e+06 | 3.783065e+06 | 6.5e-08 |
| 1 | 1.828505e+06 | 1.828505e+06 | 1.0e-07 |
| 2 | -4.7498e+03 | -4.7485e+03 | 2.8e-04 (FD-noise dominated) |

```bash
cd ~/Documents/OPM/opm_clean/opm-adjoint
./tests/run-adjoint-fd-test.sh $BIN_MOD/flow_adjoint \
    $TESTS/adjoint_tests/MODEL_1D_DEBUG/inputfiles MODEL_1D_DEBUG /tmp/fdtest
```
Note: FD runs MUST use `--enable-adaptive-time-stepping=false` — a tiny
parameter perturbation can otherwise change the substep sequence and the
discretization difference swamps the FD signal (the script handles this).

SPE1CASE1 smoke: 123-substep sweep, 0 replay failures.

### Well-objective coupling — FD-verified (all four combinations)

opm-simulators gained a slim **`adjoint-hooks` branch** (master + one
25-line commit: read accessors `Bmatrix()/Cmatrix()/Dmatrix()` on
`StandardWellEquations`) — the only opm-simulators change the module
needs, an upstream micro-PR candidate. With it, `--adjoint-objective`
selects:
- `pressure-average` (reservoir state, final substep),
- `bhp:<WELL>` (time-integrated BHP — a well primary variable, so
  dJ/dx_w is exact; the rhs gains `+B^T D^-T dJ/dx_w` per perforated
  cell, sign from Lagrangian stationarity in x_w).

| T5 combination (MODEL_1D_DEBUG) | rel. errors |
|---|---|
| pressure-average × PV | 6.5e-8 / 1.0e-7 / 2.8e-4 |
| pressure-average × trans (`run-adjoint-fd-trans-test.sh`, MULTX) | 5.7e-8 / 2.6e-8 |
| bhp:Prod × PV | **4.2e-9 / 4.2e-9 / 3.7e-9** |
| bhp:Prod × trans | 1.7e-8 / 1.5e-8 |

Gotcha encoded in the scripts: a BHP-*controlled* well has its BHP
pinned by the control equation — gradient legitimately zero; the
scripts' optional `[orat]` argument rewrites the producer to ORAT
control so the BHP is free.

### Rate and rate-matching objectives — FD-verified

The adjoint-hooks branch gained a second accessor
(`StandardWellEval::primaryVariables()`), giving the rate <->
primary-variable derivatives via `getQs()` (well-variable slots at
`numEq + j`). New objectives:
- `rate:<WELL>:<oil|water|gas>` — J = Σ dt q (cumulative volume; e.g.
  d(cumulative oil)/d(poro, trans));
- `match:<WELL>:<phase>:<target sm3/day>` — J = Σ dt (q − q_t)², **the
  quadratic well-curve-matching form** (constant target; observed
  curves via ESmry are pure I/O on top of this verified machinery).

| T5 combination (MODEL_1D_DEBUG, BHP-controlled producer) | rel. errors |
|---|---|
| rate:Prod:oil × PV | 1.0e-8 / 1.2e-8 / 6.5e-8 |
| rate:Prod:oil × trans | 8.8e-9 / 1.0e-8 |
| match:Prod:oil:10 × PV | 6.5e-9 / 7.7e-9 / 2.4e-8 |

(Control-mode gotcha mirrored: a rate objective needs the rate FREE —
i.e. a BHP-controlled well; the bhp objective needs ORAT control.)

**T0.7 update:** verified on a DRSDT variant of SPE1CASE1
(`DRSDT 0.001 ALL` in SCHEDULE): cache vs no-cache **bitwise identical**
— confirming the plan's analysis that the non-recycle branch evaluates
the identical storage expression. The spe02 radial decks and model6
(2-phase) cannot run in the 3-phase blackoil TypeTag binaries
(pre-existing limitation).

### Well-curve matching vs a reference summary (matchref) — DONE

`--adjoint-objective=matchref:<WELL>:<phase>:<refcase>`:
J = Σ dt (q(t_k) − q_obs(t_k))², observations read via `ESmry` with
linear time interpolation and unit conversion from the file metadata
(SM3/DAY, MSCF/DAY, STB/DAY — SPE1 is FIELD; the twin test caught the
missing conversion immediately). Verified on SPE1CASE1
(`tests/run-adjoint-matchref-test.sh`, reference PORO 0.285 vs base 0.3,
gas rate of the ORAT-controlled producer):
- twin check (run vs its own summary): J = 1.2e-4 vs misfit J = 5.9e8
  (12 orders; residual = summary float precision);
- FD of the global porosity scale vs Σ adjoint pvmult gradients:
  rel err 1.5e-4.

Caveat encoded in the script: replay/gradient/objective runs truncate
the summary files of their output directory at startup — keep reference
summaries outside the active output dir.

### Multi-term misfits and well adjoints

- `matchsum:<refcase>:<W>.<p>[+<W>.<p>]...` — the general weighted
  well-curve misfit over several wells/phases against one reference
  summary (rate-family objectives are now all sums of terms). Verified
  on SPE1 with PROD.gas+PROD.water and PROD.gas+PROD.oil (twin 12
  orders, FD 1.5e-4; extra terms contribute as physically expected).
- **lambda_w** computed per well per substep
  (= −D⁻ᵀ(dJ/dx_w + Σ C λ_r), Cmatrix accessor) and written to
  `<CASE>.ADJOINT_LAMBDA_WELLS.txt` — the basis for later well-control
  gradients dJ/du.

**This completes the plan's priority objective on simple blackoil
decks** (current scoping: no hysteresis / DRSDT / complex explicit
updates). Still open: producers-only restriction (injector keywords),
per-term weights from config, dJ/du via lambda_w, T6 JutulDarcy
cross-check (stage 1 = FORWARD equivalence first — see
`tests/run-jutul-forward-compare.sh`; helper library + dedicated Julia
environment in `jutul/`, see `jutul/README.md`; package precompile
pending), perm chain
rule (needs half-trans exposure from Transmissibility — small
adjoint-hooks addition, computeHalfTrans_ is private today), parts of
[#6751](https://github.com/OPM/opm-simulators/pull/6751) /
[#7039](https://github.com/OPM/opm-simulators/pull/7039) onto
adjoint-hooks when needed.

```bash
BIN_MOD=~/Documents/OPM/opm_clean/builds/release/opm-adjoint
cd ~/Documents/OPM/opm_clean/opm-adjoint
./tests/run-adjoint-replay-test.sh $BIN_MOD/flow_adjoint \
    $TESTS/spe1/SPE1CASE1.DATA /tmp/replay
```

## Historical: the in-tree implementation (branch `adjoint`, frozen)

| Repo | Branch state |
|---|---|
| **opm-simulators** | 8 commits on `adjoint` (based on master 95ab3ef44): PR1 recorder, PR2 transpose/solver, PR3 replay driver — details below |
| **opm-tests** | `adjoint` branch = master + the 2019 `adjoint_tests/` decks (MODEL_1D_DEBUG, MODEL_2D_DEBUG_3, MODEL_3D_DEBUG_1) + WCONINJE/WCONPROD VFP-item modernization |
| **opm-common / opm-grid** | `adjoint` branches unchanged (no changes needed so far) |

Milestone tags / PR-staging branches (local, opm-simulators):
`adjoint-pr1-recorder` (tag, end of PR1) = branch `adjoint-pr1`;
`adjoint-pr2-linalg` (tag) = branch `adjoint-pr2`;
`adjoint-milestone-A` (tag, HEAD) = branch `adjoint-pr3`;
opm-tests: tag `adjoint-milestone-A` on its `adjoint` branch.

Commits on opm-simulators `adjoint` (oldest first):

1. `adjoint: add runtime parameters and archive storage backend` — `--adjoint-*`
   parameters; `AdjointArchive` storing serializeOp/MemPacker buffers per
   (group, dataset) with an HDF5 backend (when built with HDF5) and a
   dependency-free directory backend; metadata schema; matrix/vector dumps +
   comparison helpers. Files under `opm/simulators/flow/adjoint/`.
2. `adjoint: record per-substep state during forward runs` — `AdjointRecorder`,
   the accepted-substep callback in `AdaptiveTimeStepping`, wiring in
   `SimulatorFullyImplicit`. Warnings-first validity guard.
3. `adjoint: add storage roundtrip unit test and T0.7 equivalence script`
4. `adjoint: add BCRS transpose utility and direct transposed-system solver` —
   `transposeBlockMatrix`, `AdjointLinearSolver` (UMFPACK on the explicit
   transpose), unit tests.
5. `adjoint: make initial snapshot replayable; add noise floor to comparison`
6. `adjoint: add backward replay driver and flow_adjoint binary` —
   `AdjointReplay` (**Milestone A complete**)
7. `adjoint: skip timestamped SMSPEC in T0.7 bitwise comparison`
8. `adjoint: run replay via --adjoint-mode=replay in flow itself` — the
   separate flow_adjoint binary is gone again (it doubled rebuild times by
   duplicating the blackoil TypeTag instantiation); **everything runs
   through `flow_blackoil`**, replay selected with `--adjoint-mode=replay`.

## Results so far

| Test | Case | Result |
|---|---|---|
| T1 storage roundtrip | unit test | PASS |
| T2/T3 backward replay | SPE1CASE1 | **123/123 substeps, fully bitwise** (worst rel diff 0.0) — directory store AND HDF5 .h5 backends, Debug and Release (-O3) builds |
| T2/T3 backward replay | MODEL_1D_DEBUG | **7/7 substeps**, bitwise above the 1e-9 noise floor (4 substeps show 1e-13–1e-11 absolute rounding noise — forward retry paths) |
| T0.7 cache equivalence | SPE1CASE1 | UNSMRY and UNRST **bitwise identical** between `--enable-storage-cache=true/false` |
| PR2 transpose + UMFPACK solve | unit test | PASS |

Known limitations / not yet done:
- The replay mode lives in the blackoil TPFA TypeTag binaries; **2-phase
  decks (e.g. SPE1CASE2_GASWATER) hit a pre-existing assert** in
  `flow_blackoil` — they need the gas-water variant later (the mode flag
  makes that automatic once `flow_gaswater`-style binaries register the
  adjoint parameters).
- Serial only; `--enable-storage-cache=false` required for record and replay.
- T0.7 on the DRSDT (`spe02`) and ROCKCOMP (`model6`) decks not yet run.
- Milestones B (adjoint solve) and C (gradients) not started; the replay
  driver contains the marked hook where `Bdiag` is computed.

## How to build

Build trees (all superbuilds of the top-level `opm_clean/CMakeLists.txt`,
Ninja, dune from `dune_installed_all`, configured with
`-C builds/cmake_minimal.cmake`):

| Tree | Build type | Notes |
|---|---|---|
| `builds/release` | Release (-O3) | recommended for running tests |
| `builds/debug`   | Debug | for debugging; slow runs |
| `builds/ninja_release` | *(empty build type — no -O flags!)* | pre-existing |
| `vscode_build` | Debug, clang | where the adjoint work was first developed and tested |

```bash
cd ~/Documents/OPM/opm_clean/builds/release   # or builds/debug
ninja flow_blackoil test_AdjointStorage test_transposeMatrix
```
Only `flow_blackoil` is needed for all forward/replay testing (fastest of
the big targets to rebuild). Binaries land in `<tree>/opm-simulators/bin/`.

**HDF5 status: parallel HDF5 active.** `hdf5-mpi` is installed and
`builds/release` / `builds/debug` are configured with `HAVE_HDF5=1`; the
default archive is now a single `CASE.ADJOINT.h5` file (browsable with
`h5dump -n`). Layout: `/adjoint/meta/0`, `/adjoint/initial/simulator/0`,
`/adjoint/substep/<k>/{simulator,wgstate,system/residual,system/jacobian}/0`.
A path without the `.h5` suffix (via `--adjoint-file`) still selects the
dependency-free directory store; both backends store identical bytes per
dataset and the replay is bitwise with either. (Historical note: serial
brew `hdf5` is rejected by MPI builds — "parallel version of hdf5"
configure warning — which is why `hdf5-mpi` is the right formula.)
`builds/ninja_release` predates the hdf5-mpi install; reconfigure it the
same way if you want `.h5` there too.
- UMFPACK (SuiteSparse) is available, so `AdjointLinearSolver` works.

## Step-by-step testing

Set for brevity:
```bash
BIN=~/Documents/OPM/opm_clean/vscode_build/opm-simulators/bin
TESTS=~/Documents/OPM/opm_clean/opm-tests
SIM=~/Documents/OPM/opm_clean/opm-simulators
```

### T1 — storage/IO unit tests

```bash
$BIN/test_AdjointStorage      # archive roundtrip, dump comparison
$BIN/test_transposeMatrix     # transpose vs umtv, UMFPACK transposed solve
```
Expected: `*** No errors detected` from both.

### T0.5 — determinism (run before blaming the replay)

```bash
for i in 1 2; do
  $BIN/flow_blackoil $TESTS/spe1/SPE1CASE1.DATA --output-dir=/tmp/det$i \
      --threads-per-process=1 --enable-storage-cache=false
done
cmp /tmp/det1/SPE1CASE1.UNSMRY /tmp/det2/SPE1CASE1.UNSMRY && echo DETERMINISTIC
```

### T0.7 — storage cache vs no-cache equivalence

```bash
cd $SIM
./tests/run-storage-cache-equivalence.sh $BIN/flow_blackoil \
    $TESTS/spe1/SPE1CASE1.DATA /tmp/t07
```
Expected: `UNSMRY: bitwise identical`, `UNRST: bitwise identical`, PASS.

Still to be run (expected **bitwise**, since recycling is disabled there and
both modes evaluate the identical storage expression):
```bash
./tests/run-storage-cache-equivalence.sh $BIN/flow_blackoil \
    $TESTS/spe02/SPE02-RADIAL-02.DATA /tmp/t07_drsdt        # DRSDT
./tests/run-storage-cache-equivalence.sh $BIN/flow_blackoil \
    $TESTS/model6/0B_ROCKTAB_MODEL6.DATA /tmp/t07_rockcomp  # ROCKCOMP
```
Any deviation beyond rounding here is a genuine storage-term bug worth
reporting upstream independently of the adjoint work. With a build of
opm-common's `compareECL` you can pass it as 4th argument for tolerance-based
comparison instead of bitwise.

### T2/T3 — backward replay (Milestone A acceptance)

One-liner via the regression script:
```bash
cd $SIM
./tests/run-adjoint-replay-test.sh $BIN/flow_blackoil \
    $TESTS/spe1/SPE1CASE1.DATA /tmp/replay_spe1
./tests/run-adjoint-replay-test.sh $BIN/flow_blackoil \
    $TESTS/adjoint_tests/MODEL_1D_DEBUG/inputfiles/MODEL_1D_DEBUG.DATA /tmp/replay_m1d
```
Or manually, which shows the moving parts:
```bash
# 1. forward run with per-substep recording + stored systems
$BIN/flow_blackoil $TESTS/spe1/SPE1CASE1.DATA --output-dir=/tmp/adj \
    --adjoint-save=true --adjoint-save-system=true \
    --enable-storage-cache=false --threads-per-process=1
# -> /tmp/adj/SPE1CASE1.ADJOINT/  (or .h5 with HDF5)
#    adjoint/initial/simulator.bin            initial snapshot
#    adjoint/substep/<k>/simulator.bin        full state at end of substep k
#    adjoint/substep/<k>/wgstate.bin          well/group state alone
#    adjoint/substep/<k>/system/{residual,jacobian}.bin
#    adjoint/meta.bin, adjoint/parameters

# 2. backward replay + comparison (same binary, replay mode)
$BIN/flow_blackoil $TESTS/spe1/SPE1CASE1.DATA --output-dir=/tmp/adj \
    --adjoint-mode=replay \
    --enable-storage-cache=false --threads-per-process=1
```
Per-substep log lines (PRT/DBG and stdout):
```
replay substep  122 (report step 119, dt 31 days): residual rel diff 0.000e+00 (abs 0.000e+00), Jacobian rel diff 0.000e+00 (abs 0.000e+00)
...
Adjoint replay finished: 123/123 substeps within tolerance 1.0e-12 (...)
```
Exit code 0 = every substep within tolerance. Tolerances:
`--adjoint-replay-tolerance` (relative, default 1e-12) and
`--adjoint-replay-abs-tolerance` (absolute noise floor, default 1e-9 —
entry differences below it are rounding noise and excluded from the
relative criterion).

What the replay does per substep k (see `AdjointReplay::replayStep`):
restore snapshot k−1 → mirror the report-step context
(`startNextEpisode`/`beginEpisode`) → mirror `prepareStep`
(`advanceTimeLevel`, `setTime`, `setTimeStepSize`, `beginTimeStep`) →
iteration-0 assembly (well `prepareTimeStep` explicit quantities from state
k−1, exactly like forward) → install x_k + WellState_k → final linearization
→ compare with stored system. Afterwards the linearizer holds the converged
system of substep k — the state Milestone B's adjoint solve starts from.

### Interpreting failures

- `WellState size mismatch` on reading a snapshot: archive recorded with an
  older commit of the branch — re-run the forward simulation.
- Residual rel diff large but abs diff ~1e-12: rounding noise from forward
  retry (chopped step) paths; covered by the abs-tolerance floor.
- Genuine structural mismatch / large diffs on decks with group control,
  THP/VFP, mid-episode control switching: this is plan risk #1 (well
  re-assembly fidelity). Restrict to plain BHP/rate controls for now; the
  fallback (`assembleWellEqWithoutIteration`) is the next thing to try.

### Forward-run validity warnings

With `--adjoint-save=true` the recorder warns (warnings-first guard) when:
the storage cache is enabled, more than 1 MPI rank, more than 1 thread,
system dumps disabled. The warning list is the worklist from the plan's
catalog of explicitly-updated quantities.

## Next steps (per the plan)

1. **T0.7 on spe02/model6**, and replay tests on MODEL_2D_DEBUG_3 /
   MODEL_3D_DEBUG_1 (multi-well 3D; exercises the well fidelity risk).
2. **Milestone B**: Bdiag at the marked hook in `replayStep` (cell-local
   `computeStorage<Evaluation>` at IQ(1)); `WellCurveObjective` (ESmry vs
   reference UNSMRY); adjoint recursion with `AdjointLinearSolver`;
   then the [#6751](https://github.com/OPM/opm-simulators/pull/6751)
   full-system path when merged.
3. **Milestone C**: PV/trans multiplier gradients accumulated during the
   sweep; T4 (1-cell sign test) and T5 (FD verification on MODEL_1D_DEBUG).
4. Rebase onto [#7039](https://github.com/OPM/opm-simulators/pull/7039)
   when it lands (exact forward replay semantics for failed updates).
5. ctest integration of the two shell scripts (currently run manually).
