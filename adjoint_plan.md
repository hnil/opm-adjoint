# Adjoint capability for OPM flow — assessment of the 2019 prototype and implementation plan

*Draft for discussion — June 2026. Halvor M. Nilsen (with Claude Code assistance).*


## Context

Goal: gradients of well-curve-matching objectives (well rates/BHP vs. observed) with respect to
porosity and permeability/transmissibility, computed by the adjoint method in OPM flow; later
extensions to end-point scaling and (the original 2019 goal) well-rate control optimization.

Previous attempt (June-2019 codebase, **not rebasable** — predates the opm-models merge into
opm-simulators, the removal of `ebos/`/`opm/autodiff/`, and the TpfaLinearizer):

- `hnil/opm-simulators @ adjoint_new_try` (~2800 lines): `runAdjoint()` backward loop in
  `SimulatorFullyImplicitBlackoilEbos`, per-step serialization (ewoms restart + boost archives for
  well state), re-linearization in the backward pass, transposed full matrix solved with
  `Dune::UMFPack`, well-side adjoint (`applyt()`, `λ_w = D^{-T}(C^T λ_r + ∂J/∂w)`, control
  derivatives via `duneCA_`/`duneDA_`), rate-based objective functions
  (`opm/objective/ObjectiveFunctions.hpp`). Last commit: "tests still fail".
- `hnil/opm-models @ adjoint_new_try`: threaded a `focusTimeIdx` through
  linearizer → elementContext → `makeEvaluation` → intensive quantities so AD could differentiate
  the residual w.r.t. *old-time* primary variables; fluxes assumed purely implicit so only the
  storage term carries old-time derivatives.
- `hnil/opm-common @ adjoint_devel_new / add_adjoint`: essentially only cmake prereqs — nothing to port.
- `hnil/opm-tests @ add_adjoint_tests`: decks + reference `adjoint_results` for
  `adjoint_tests/{MODEL_1D_DEBUG, MODEL_2D_DEBUG_3, MODEL_3D_DEBUG_1}` — reusable as-is.
- Parameter derivatives (poro/perm) were **never implemented** in the prototype; it was
  well-control focused.

What survives as *design*: the backward recursion, the well Schur-complement adjoint algebra, the
objective-function shape, UMFPack-on-transpose as first solver. What is obsolete: the serialization
approach (boost + ewoms ascii restart → replaced by master's HDF5 `serializeOp` machinery), the
timer reverse-stepping hack, and the full `focusTimeIdx` threading (the TpfaLinearizer's
storage/flux split makes `dR_{n+1}/dx_n` a cell-local block-diagonal computation).

## Key facts about current master (verified)

1. Default blackoil flow uses the **TpfaLinearizer** (`flowBlackoilTpfa*Main` in
   `opm/simulators/flow/MainDispatchDynamic.cpp::runBlackOil`); `FvBaseLinearizer` only for the
   diffusion fallback. Adjoint instrumentation targets `opm/models/discretization/common/tpfalinearizer.hh`.
2. In `tpfalinearizer.hh::linearize_cell`, flux and source use only timeIdx-0 intensive
   quantities; the accumulation term is `storage(x_n) − cachedStorage(x_{n-1})` scaled by `V/dt`.
   ⇒ **dR_{n+1}/dx_n = −(V/dt)·∂storage/∂x|x_n is block-diagonal and cell-local** — computable by a
   standalone loop calling `LocalResidual::computeStorage<Evaluation>`; no opm-models threading needed.
3. Assembly sequence to reproduce (`NonlinearSystemBlackOilReservoir_impl.hpp::assembleReservoir`):
   `problem().beginIteration()` → `BlackoilWellModel::assemble(dt)` →
   `linearizer().linearizeDomain()` → `problem().endIteration()`. At convergence the linearizer
   holds the converged residual/Jacobian *without* the well Schur complement; each well's
   `StandardWellEquations` (`duneB_`, `duneC_`, `duneD_`, `invDuneD_`, `resWell_`) holds the
   converged well system.
4. Storage-cache subtlety: at global iteration 0 the cache for timeIdx 1 is filled (recycled from
   timeIdx 0 when `recycleFirstIterationStorage()`); replay therefore needs a two-linearization
   sequence (first at x_{k-1}, then at x_k) to reproduce the forward cache state exactly.
5. Serialization: `--save-step/--load-step` already snapshots simulator (model solutions at both
   time indices, problem history incl. `drift_`, hysteresis, `mixControls_`), WellState/GroupState
   (WGState), aquifers, time/episode via `serializeOp` + `HDF5Serializer`
   (`opm/simulators/utils/HDF5Serializer.hpp`). Missing for adjoint: per-substep history,
   residual/Jacobian dumps, substep metadata.
6. The substep loop lives in `AdaptiveTimeStepping<TypeTag>::SubStepIteration::run()`
   (`opm/simulators/timestepping/AdaptiveTimeStepping_impl.hpp`); accepted substeps are the natural
   recording hook (after `problem.endTimeStep()`).
7. No transpose-solve support in ISTL solvers; `Dune::UMFPack` available behind
   `HAVE_SUITESPARSE_UMFPACK` (cf. `MSWellHelpers.cpp::applyUMFPack`). Port 2019 `transpose.hh`.
8. `StandardWellEquations::extract()` adds `A −= CᵀD⁻¹B` to the global matrix — works in the TPFA
   path via `WellConnectionAuxiliaryModule`; since `(A − CᵀD⁻¹B)ᵀ = Aᵀ − BᵀD⁻ᵀC`, the adjoint can
   use the Schur-reduced matrix plus per-well transposed recovery — no monolithic coupled matrix needed.

## Reference implementations: JutulDarcy (most relevant) and MRST

**Jutul.jl** (`src/ad/gradients.jl`, SINTEF; JutulDarcy paper: Computational Geosciences
10.1007/s10596-025-10366-6) implements exactly the recursion we plan, which both validates the
design and gives naming/structure to align with:

- `solve_adjoint_sensitivities(model, states, timesteps/reports, G)`: backward loop
  `for i in N:-1:1`, per step a state triplet `(s0, s, s_next)` from an `AdjointPackedResult`
  (all forward states kept **in memory**; MRST uses `ResultHandler` to spill states to disk —
  our HDF5 file plays that role, with an in-memory option later).
- Per step: `adjoint_reassemble!(sim, state, state0, dt, forces, time)` re-linearizes at the
  stored states (same choice as ours: re-linearize, don't store matrices — our stored
  system dumps are verification extras only); the adjoint solve uses **the same linear system as
  forward, transposed**; `next_lagrange_multiplier!` builds
  `rhs = −(∂J/∂xₙ)ᵀ − (∂Fₙ₊₁/∂xₙ)ᵀ λₙ₊₁` via matrix-free `sens_add_mult!`.
- **Parameter derivatives**: parameters are promoted to AD variables of an auxiliary "parameter
  model" (`adjoint_parameter_model` / `swap_primary_with_parameters!`,
  `parameter_is_differentiable`), and a dedicated simulator in `mode=:sensitivities` assembles
  `∂Fₙ/∂p`; gradient accumulated as `∇G += (∂Fₙ/∂p)ᵀ λₙ` (`update_sensitivities!`), plus the
  direct `∂J/∂p` term. Initial-state gradients via `update_state0_sensitivities!`.
- **Objective interface**: sum-type objectives `G(model, state, dt, step_info, forces)` per
  (sub)step, `step_info` carrying time/dt/report-step/substep indices; also global objectives
  over the whole trajectory.

**MRST** (`ad-core/simulators/computeGradientAdjointAD.m`): same backward structure
(`model.solveAdjoint` per step, states from memory or disk via `ResultHandler`), plus one detail
worth copying: **control-equation scaling** (`getControlEqScaling`) to normalize well-constraint
rows (BHP vs rate magnitudes) before the transposed solve — relevant for the conditioning of our
`D^T` solves and the FD comparison.

Consequences adopted in this plan:
1. Same recursion and re-linearize-backward strategy (already planned) — keep our HDF5 store as
   the "ResultHandler" equivalent; add a keep-in-memory option later as an optimization.
2. Objective-function interface copied from Jutul: per-substep
   `J_n(simulator, wellState, dt, stepInfo)` sum objectives, so OPM/Jutul objectives stay
   comparable for cross-validation (JutulDarcy reads OPM decks — the same case can be run in both
   and gradients diffed, an extra verification channel beyond finite differences).
3. For porosity/PV and transmissibility v1 we keep the *analytic* dR/dp (storage linear in PV,
   TPFA flux linear in T — simpler and exact). The Jutul "parameters as AD variables" pattern is
   the designated route for the **end-point-scaling extension**: evaluate storage+flux cell/face
   contributions with an `Evaluation` type whose derivative slots are local parameters (a
   parameter-AD pass over `linearize_cell`'s ingredients), instead of threading new variables
   through the whole model.

## Prerequisites — existing PRs to merge first

1. **[opm-simulators#7039](https://github.com/OPM/opm-simulators/pull/7039)** "Comparing fixed
   timestep only + failed update" (+ companion opm-common branch `fix_replay`): makes forward
   replay with recorded timesteps "exact up to rounding", refactors `updateFailed`/
   `advanceTimeLevel` and failure resetting. **Milestone A builds directly on this** — merge (or
   rebase the adjoint work onto it) before starting the recorder/replay code, and reuse its
   model2-variant replay verification as test level T0 below.
2. **[opm-simulators#6751](https://github.com/OPM/opm-simulators/pull/6751)** "Linear solver
   system" (hnil + jakobtorben, ready for review): solves the **full coupled reservoir+well
   system** — well DOFs in the global matrix (well-matrix merger, SystemMatrix views,
   CPR-equivalent preconditioner). Adopted as the basis for the adjoint solve (Milestone B):
   transpose the monolithic matrix once per step instead of implementing transposed Schur
   recovery (`D^{-T}`, `applyt`) in the well code. The Schur-transpose path from the 2019
   prototype is demoted to fallback if #6751 stalls.
3. **`cprt` already exists in master**: `PreconditionerFactory` registers `"cprt"` (serial, MPI,
   GPU factories — `StandardPreconditioners_serial.hpp:246` etc.), built on
   `PressureTransferPolicy<..., transpose=true>` and `getQuasiImpesWeights(..., transpose=true)`,
   i.e. CPR for the **transposed** system. No new preconditioner work is needed for an iterative
   adjoint solve — select it via FlexibleSolver JSON config.

## Run-configuration guarantees

The adjoint is only as good as the runs feeding it, so the design enforces four guarantees:

1. **Validity guard on forward runs — warnings first**: when `--adjoint-save` is on,
   `AdjointRecorder` checks the run configuration at startup and **warns loudly** (log, plus a
   flag recorded in the HDF5 header) about anything known to threaten exact replay or gradient
   quality: relaxed CNV acceptance enabled, loose `--tolerance-cnv/--tolerance-mb`, features
   outside the tested envelope (MSW, networks, aquifers for gradient runs), active
   explicitly-updated quantities from the catalog below. Initially nothing is refused — the
   warning list *is* the worklist; individual checks are promoted to hard errors only once a
   feature is demonstrated broken (or supported and removed from the list). The complete
   parameter set is written into the HDF5 header (existing `HDF5Serializer::writeHeader`
   pattern), and the replay/adjoint stage compares it against its own configuration and warns on
   mismatch.
2. **Replay always from recorded timesteps**: the backward pass exclusively traverses the
   recorded accepted substeps (`/adjoint/meta` dt sequence); a forward *re*-run, when needed, uses
   the PR #7039 fixed-timestep mode driven by the same recorded sequence. No adaptive decisions
   are ever taken in replay/adjoint stages.
3. **Same-binary, same-process option**: `flow_adjoint` supports a one-stop mode (forward sweep →
   immediately backward sweep in the same process) in addition to the two-stage
   save-then-replay mode. One-stop is the default for gradient computation — it rules out
   binary/config mismatch between stages and allows a later optimization of keeping states in
   memory (Jutul-style `AdjointPackedResult`) instead of round-tripping through HDF5; two-stage
   stays as the debugging and test workflow (T2/T3 need the on-disk systems anyway).
4. **Determinism before replay**: a test that runs the same forward case twice (pinned
   `--threads-per-process=1`, single rank, fixed partitioning) and requires bitwise-identical
   summary/restart output sits *below* the replay tests in the ladder (T0.5) — if determinism is
   broken, T2/T3 failures would otherwise be misattributed to the adjoint machinery.

### Catalog of explicitly-updated quantities (need particular treatment)

A number of quantities are updated *explicitly* between timesteps (mostly in
`FlowProblem`/`FlowProblemBlackoil` `beginTimeStep()`/`endTimeStep()`); they are exactly the
places where replay can silently diverge and where the adjoint recursion has hidden cross-terms.
An early task (start of PR1, revisited at PR3 and PR4) is an audit pass over
`FlowProblem::endTimeStep`/`beginTimeStep` and the well model producing a table — for each
quantity: (a) replay treatment, (b) adjoint treatment, (c) warning condition. Initial catalog:

| Quantity | Where updated | Replay (Milestone A) | Adjoint (Milestones B/C) |
|---|---|---|---|
| Hysteresis state (max/min saturations in `materialLawManager_`) | `updateHysteresis_()` in `endTimeStep` | serialized in problem snapshot ⇒ exact | cross-term dR/d(hyst state) neglected v1 ⇒ **warn when hysteresis active**; nohyst variant preferred for gradients |
| Mixing controls DRSDT/VAPPARS (`mixControls_` max values) | `endTimeStep` | serialized (`FlowProblemBlackoil::serializeOp`) ⇒ exact | neglected cross-term ⇒ warn when active; also disables storage-cache recycling (replay handles, gradient test pending) |
| Max oil saturation / max water saturation tracking (VAPPARS, rock compaction min pressure) | `updateMaxOilSaturation_` etc. in `endTimeStep` | serialized ⇒ exact | neglected ⇒ warn |
| Drift compensation `drift_` | end of step | serialized ⇒ exact | recommend `--enable-drift-compensation=false` for gradient runs; warn otherwise |
| Polymer adsorption / max polymer | `endTimeStep` | serialized ⇒ exact | out of v1 scope; warn |
| Tracer state | tracer model `endTimeStep` | serialized ⇒ exact | not in objective v1; no cross-term needed if tracers don't feed back |
| Aquifer internal state | aquifer `endTimeStep` | serialized ⇒ exact | neglected ⇒ exclude aquifer decks from gradient tests; warn |
| Well explicit quantities (`F0_` wellbore storage, connection pressures, explicit B-factors) | `prepareTimeStep`/`calculateExplicitQuantities` at `beginTimeStep` | recomputed from state k−1 in replay step 3 ⇒ exact | known neglected cross-term (documented v1 approximation); FD tests quantify; fixable via well-level storage-derivative trick |
| Well/group control switching state | during Newton + `beginTimeStep` | WGState snapshot ⇒ exact (final converged controls) | control-active pattern frozen per step in adjoint; kinks at switches are real (objective nonsmoothness) — note in docs, not a bug |
| Newton-meta (relaxation, chopped dt history) | AdaptiveTimeStepping | recorded in `/adjoint/meta` | n/a (only accepted steps enter the adjoint) |

The catalog is also the natural reviewer-facing artifact for upstream discussion: it makes
explicit what "exact replay" covers and what the v1 gradient knowingly ignores.

## Test ladder (named levels, used as milestone acceptance criteria)

- **T0 — perfect forward replay** (from PR #7039): re-run with recorded substep sizes reproduces
  the forward simulation up to rounding (model2 variants).
- **T0.5 — determinism**: identical forward run twice (1 thread, 1 rank) ⇒ bitwise identical
  output; isolates nondeterminism failures from replay failures.
- **T0.7 — storage-path equivalence (cache vs no-cache)**: the same deck run with
  `--enable-storage-cache=true` and `=false` (1 thread, 1 rank) must agree. Two tiers:
  (a) *system level, zero code needed, runnable today*: compare summary + restart output on
  (i) SPE1CASE1 and a model2 variant (recycling valid — expect agreement to rounding; report the
  max deviation), (ii) an `opm-tests/spe02` deck (DRSDT active ⇒ non-recycle branch — expect
  **bitwise** agreement, since cache and no-cache evaluate the identical expression), and
  (iii) `model6/0B_ROCKTAB_MODEL6` (rock compaction, also non-recycle). Any deviation beyond
  rounding in (i) or any deviation at all in (ii)/(iii) is a genuine bug in the storage-term
  handling — worth reporting upstream independently of the adjoint work.
  (b) *linear-system level, after PR1*: one report step in each mode with
  `--adjoint-save-system=true`, diff the converged residual/Jacobian dumps directly — this is
  the form that certifies gradients are independent of the cache mode and gates re-enabling the
  cache as an adjoint optimization later.
- **T1 — IO roundtrip** (unit, ctest): adjoint HDF5 store — BCRS matrix, residual, well matrices,
  snapshots through `serializeOp` — write/read identity.
- **T2 — perfect re-linearization of one step**: from stored snapshots, recreate residual AND
  Jacobian (reservoir + per-well B,C,D,resWell) of a single converged forward substep; compare to
  the stored system, target bitwise with `--threads-per-process=1`, formal pass 1e-12 relative.
- **T3 — perfect backward propagation** (Milestone A acceptance): full reverse sweep over all
  accepted substeps with the T2 check at every step (SPE1CASE1, SPE1CASE2_GASWATER,
  MODEL_1D_DEBUG; later a DRSDT deck).
- **T4 — adjoint sanity**: 1-cell/1-well/2-step case; trivial objectives (e.g. J = sum of well
  rate at last step) where λ and dJ/dm can be checked by hand; pins all sign conventions.
- **T5 — gradient vs finite differences** (Milestones B+C acceptance): central FD on MULTPV/PERM
  multipliers, cell-wise on MODEL_1D_DEBUG, sampled cells on SPE1; rel-tol ~1e-4 (FD-noise bound).
- **T6 — cross-validation vs JutulDarcy** (scripted, not CI): same deck + same objective through
  `solve_adjoint_sensitivities`; expect close agreement on simple TPFA decks with plain
  rate/BHP-controlled wells.

**JutulDarcy vs FD for testing — recommendation:** both, in different roles. FD is the
authoritative oracle (same binary, catches everything end-to-end including deck input handling)
and is what goes into CI (T5) — but it is noisy (step-size choice) and O(#params) expensive, so
only on tiny decks. JutulDarcy is the *development* instrument: an independent adjoint on the same
deck, no FD noise, ideal for locating sign/scaling/convention bugs during Milestone B bring-up and
for medium-sized cases where FD is unaffordable — but it stays out of CI (Julia dependency, and
physics/well-model differences put a floor on agreement except for the simplest decks).

## Plan

### Milestone A — exact backward replay (priority 1, includes tests)

**Storage (write too much, optimize later):** one HDF5 file per run (`<CASE>.ADJOINT.h5`) via the
existing `HDF5Serializer`/`serializeOp`, keyed by global accepted-substep index k:

```
/adjoint/meta                      schema version, numSubsteps,
                                   per-substep {globalIdx, reportStep, substepInReport,
                                   startTime, dt, newtonIterations}, cell mapping
/adjoint/substep/<k>/simulator     full simulator snapshot at END of substep k
                                   (solution(0)=x_k, solution(1)=x_{k-1}, problem history, WGState)
/adjoint/substep/<k>/wgstate       well/group state alone (so replay can combine snapshot k-1
                                   reservoir state with WellState_k)
# verification extras, --adjoint-save-system:
/adjoint/substep/<k>/system/{residual, jacobian}     converged R and BCRS dump
/adjoint/substep/<k>/wells/<name>/{B,C,D,resWell}    converged well system + primary vars + controls
/adjoint/substep/-1/simulator      initial condition
```

**New files** (all `HAVE_HDF5`-guarded, compiled out of default flow):
- `opm/simulators/flow/adjoint/AdjointParameters.hpp/.cpp` — `--adjoint-save`, `--adjoint-file`,
  `--adjoint-save-system`, `--adjoint-mode`, `--adjoint-objective-file`, `--adjoint-replay-tolerance`
- `opm/simulators/flow/adjoint/AdjointRecorder.hpp` — recording during forward run
- `opm/simulators/flow/adjoint/AdjointSystemIO.hpp` — BCRS/well-matrix ↔ HDF5
- `opm/simulators/flow/adjoint/AdjointReplay.hpp` — backward loop + comparison
- `flow/flow_adjoint.cpp` — `flow_adjoint` binary **in the normal flow path** (user preference):
  reuses the standard blackoil TPFA TypeTag and `flowBlackoilTpfaMainInit` plumbing from
  `flow/flow_blackoil.cpp` (same `FlowMain`/vanguard/eclState setup as production flow), but runs
  `AdjointReplay`/`AdjointSolver` instead of `SimulatorFullyImplicit::run()`. The recorder side is
  in any case inside normal flow (`--adjoint-save` on the standard `flow` binary). Fallback only
  if the `FlowMain` wiring fights back: same driver as a flowexperimental target first, moved to
  `flow/` once stable.

**Forward hooks (small diffs):** `acceptedSubstepCallback_` (std::function, mirroring the existing
`tuningUpdater` pattern) in `AdaptiveTimeStepping`, invoked after `problem.endTimeStep()` in the
accepted branch; recorder owned by `SimulatorFullyImplicit` (also covers the non-adaptive branch).

**Storage-term decision — run adjoint without the storage cache (v1), with a cache-equivalence
test guarding both modes.** The relevant code facts (verified):
- `recycleFirstIterationStorage()` (`FlowProblemBlackoil.hpp:952`) is false exactly when
  DRSDT/DRVDT are active or rock-compaction poro multipliers are present — the cases where mass
  computed from primary variables changes across the step boundary due to explicitly-updated
  history. (Hysteresis is deliberately *not* in this list: it affects relperm/flux, not mass.)
- Cache mode with recycling (`tpfalinearizer.hh::linearize_cell`): cache[1] is the storage value
  recomputed at iteration 0 of the new step from IQ(0) at x_{k-1} — fast, IQ history size 1, but
  carries the assumption "nothing changed affecting masses computed from primary variables", and
  the cache update is a state machine (gated on `isFirstGlobalIteration()`, NLDD stale-cache
  caveats).
- Cache mode without recycling (DRSDT etc.): cache[1] = `computeStorage<Scalar>(IQ(globI,1))`
  evaluated once at iteration 0 — **the identical formula the no-cache branch evaluates every
  iteration**; IQ history size becomes 2 and `shiftIntensiveQuantityCache` copies converged
  IQ(0)→IQ(1), so the old mass equals the previous step's converged mass (telescoping mass
  conservation with the *old* history state).
- No-cache branch: per-iteration `computeStorage<Scalar>(IQ(1))`; the only path used on GPU, so
  maintained and exercised. Requires IQ history 2 (`intensiveQuantityHistorySize()` already
  returns 2 when the cache is off; the guard at `tpfalinearizer.hh:875` enforces IQ availability).

Hence: **under DRSDT/VAPPARS, cache and no-cache coincide by construction** (frozen-at-iteration-0
vs recomputed value of the same expression); the assumption-carrying branch is *recycling*, which
OPM itself disables exactly where it would be wrong. Forcing no-cache is therefore assumption-free
everywhere and never changes the intended old-mass value: where recycling is valid the difference
is at most rounding (Evaluation- vs Scalar-arithmetic, evaluation context), where it is not valid
the two paths are already identical.

Decision for v1: adjoint-save and replay modes force `--enable-storage-cache=false` (existing
parameter). Consequences:
- replay simplifies to a **single linearization per substep**; the two-pass cache-reproduction
  choreography is only needed for the later cache-enabled optimization;
- cost: one extra `computeStorage` per cell per Newton iteration (IQ(1) cached — cheap) plus
  memory for one extra IQ level per cell; acceptable for serial v1, and cache mode remains the
  memory-light option for large models, re-enabled later and certified by the same T2/T3 tests
  plus the T0.7 equivalence test;
- the adjoint cross-term `Bdiag` is **identical in both modes** (derivative of the same
  `computeStorage` at x_n), so this is purely a replay-robustness choice, not a change to the
  gradient math;
- warning-list item: no-cache trajectories may differ from production-default runs at rounding
  level when recycling was valid (T0.7 quantifies this);
- week-1 spike (zero code needed): T0.7 tier (a) below is runnable today.

**Replay sequence per substep k** (mirrors `prepareStep`/`assembleReservoir`; v1 no-cache mode):
1. deserialize snapshot k−1; set episodeIndex/time/dt from meta[k]
2. `model().advanceTimeLevel()`; `problem().resetIterationForNewTimestep()`; `problem().beginTimeStep()`
   (runs well `prepareTimeStep` → explicit quantities from state k−1, exact); IQ(1) now reflects x_{k-1}
3. compute `Bdiag_k = −(V/dt)∂storage/∂x|x_{k-1}` from IQ(1) (needed in Milestone B)
4. install x_k into solution(0), deserialize WellState_k, `invalidateAndUpdateIntensiveQuantities(0)`
5. single linearization (`beginIteration` → `linearizeDomain` → `endIteration`); fallback if well
   control logic drifts: `BlackoilWellModel::assembleWellEqWithoutIteration(dt)`
6. compare residual/Jacobian/well matrices against stored; report max relative error

(For the later cache-enabled optimization, step 3 becomes the iteration-0 linearization that
reproduces the forward cache state — the two-pass sequence from the original design.)

No backward timer needed — each substep is self-described (kills the 2019 `operator--` hack).

**Tests:** unit roundtrip test for `AdjointSystemIO`; regression script
`tests/run-adjoint-replay-test.sh` (forward with `--adjoint-save --adjoint-save-system
--enable-storage-cache=false --threads-per-process=1`, then `flow_adjoint --adjoint-mode=replay`,
tolerance 1e-12 rel / target
bitwise) on SPE1CASE1, SPE1CASE2_GASWATER and `adjoint_tests/MODEL_1D_DEBUG` (merge
`hnil/opm-tests add_adjoint_tests` into the opm-tests checkout).

### Milestone B — adjoint solve (objective on well DOFs)

**Primary path (using PR #6751 full-system solve):** assemble the monolithic coupled matrix
`M = [[A, Cᵀ],[B, D]]` with the well-matrix merger from #6751, transpose it explicitly, solve

```
Mᵀ [λ_r; λ_w]_k = [ −(dJ_k/dx_r)ᵀ − Bdiag_{k+1}ᵀ λ_{r,k+1} ;  −(dJ_k/dxw)ᵀ ]
```

— the well-curve objective enters the rhs directly in the well rows; no transposed-Schur recovery
code in the well classes at all. Apply MRST-style control-equation scaling to the well rows first.
Fallback path (if #6751 stalls): Schur-reduced recursion as in the 2019 prototype
(`A_redᵀ λ_r = rhs`, then per-well `λ_w = D^{-T}((dJ/dxw) − C λ_r)`).

**Linear solver decision — explicit transpose first, no immediate investment in native
transposed-operator solves:** transposing a BCRS matrix is one structural pass + a copy per
substep, negligible next to the solve, and it makes every existing solver/preconditioner usable
unchanged. Concretely:
- v1 (tests, small/medium decks): explicit `Mᵀ` + `Dune::UMFPack` (direct — removes one error
  source while validating the recursion).
- v1.5 (already available, no new code): explicit `Mᵀ` + FlexibleSolver with the existing
  **`cprt`** preconditioner (transposed quasi-IMPES weights + transposed transfer,
  `StandardPreconditioners_serial.hpp`/`_mpi.hpp`) + BiCGSTAB/GMRES, selected via the JSON
  linear-solver config. An early spike should benchmark cprt-on-Mᵀ vs ILU0-on-Mᵀ on SPE1/model2.
- Later only if memory/GPU demands it: matrix-free transposed operator (`mtv`-based) with a
  native transposed preconditioner.

**Solver accuracy when adjoint is the aim** (applies to forward AND adjoint runs):
- The gradient error inherits the *forward* convergence error: λ weights the converged residuals,
  so steps accepted under relaxed tolerances (`--tolerance-cnv-relaxed`, time-step chopping
  acceptance) poison the gradient. Adjoint-quality forward runs need a strict mode: tightened
  `--tolerance-cnv`/`--tolerance-mb` (~1e-6/1e-7), relaxed acceptance disabled, tight linear
  reduction — exactly what the 2019 `runTestAdjoint.py` flags did. Document a recommended
  `--adjoint-strict` option set registered with the adjoint parameters.
- Adjoint solves: λ accuracy multiplies dR/dm directly; use a tighter reduction than forward
  (≤1e-10) or the direct solver for tests, and judge convergence on the true residual norm of the
  transposed system (CNV-style scaled criteria don't apply).
- FD verification tolerance must be set consistently with these (T5's 1e-4 assumes the strict
  forward mode).

Components: `opm/simulators/linalg/transposeMatrix.hpp` (port of 2019 `transpose.hh` + unit test);
`adjoint/AdjointLinearSolver.hpp` (UMFPack v1 + FlexibleSolver/cprt option behind one interface);
`adjoint/WellCurveObjective.hpp` — J = Σ w(q−q_obs)², observed data read from a reference run's
UNSMRY via `Opm::EclIO::ESmry` + keyword config; per-substep sum-objective interface modeled on
Jutul's `G(model, state, dt, step_info, forces)`; dJ/dxw via EvalWell on
`StandardWellPrimaryVariables` (2019 `ObjectiveFunctions.hpp` is the reference design); apply
MRST-style control-equation scaling before the transposed well solves;
`adjoint/AdjointSolver.hpp` (orchestration); additions to `StandardWellEquations` (B/C/D getters,
`solveTransposed`) only if the Schur fallback path is needed.

With the full-system path λ_w is computed directly as part of the monolithic solve, which keeps
the door open for later well-control gradients dJ/du (2019 duneCA_/duneDA_ design) without extra
machinery.

**First input format (objective + parameter specification):** one JSON config per case,
`--adjoint-config=<CASE>_ADJOINT.json` — JSON because the FlexibleSolver linear-solver config
already establishes the pattern (PropertyTree parsing exists in opm-simulators). Sketch:

```json
{
  "objective": {
    "type": "well_curve_match",
    "reference_summary": "path/to/REFCASE",      // read via Opm::EclIO::ESmry
    "terms": [
      { "keyword": "WOPR", "wells": ["PROD1","PROD2"], "weight": 1.0, "scale": "auto" },
      { "keyword": "WBHP", "wells": ["INJ1"], "weight": 0.1 }
    ],
    "time_weighting": "dt",                       // integrate (q − q_obs)² dt
    "window": { "start_step": 0, "end_step": -1 }
  },
  "parameters": [
    { "name": "PORVMULT",  "kind": "per_cell" },   // multiplier semantics = MULTPV
    { "name": "TRANSMULT", "kind": "per_connection" } // multiplier on face trans
  ],
  "linear_solver": { "type": "umfpack" },          // or FlexibleSolver JSON with "cprt"
  "output": { "gradients": "hdf5+text" }
}
```

Rationale for starting with *multiplier* parameters (PORV and trans multipliers, value 1.0):
gradients map 1:1 to FD perturbations of MULTPV/MULTX-style deck keywords, they're the natural
parameterization for assisted history matching, and dJ/dPERM* comes later as pure post-processing
(chain rule through `Transmissibility`) without changing the interface. Objective value per
substep and total J are reported to the log and stored next to the gradients.

Documented v1 approximation: cross-step dependence of well equations through explicit quantities
(F0_, connection pressures) neglected; quantified by FD tests; can be added later with the same
storage-derivative trick at well level.

### Flexible parameter linearizer (design, added after Milestone A landed)

**Problem:** the plan's analytic dR/dp covers PV/trans multipliers, but the general mechanism —
evaluating the cell residual with derivatives w.r.t. *chosen parameters* (poro, trans, perm,
later end-point scaling) — needs an architecture. There is no standalone fully-templated
"evaluate residual of one cell" function today; the AD type and parameter values are baked into
the model's IntensiveQuantities.

**Code reality (verified, much better than 2019):** the GPU refactoring already made the TPFA
residual kernels static and templated on *all* relevant types —
`BlackOilLocalResidualTPFA::computeStorage<LhsEval, StorageType, IntensiveQuantitiesType>` and
`computeFlux/calculateFluxes_<RateVectorT, IntensiveQuantitiesT, ResidualNBInfoT, ModuleParamsT>`
(`opm/models/blackoil/blackoillocalresidualtpfa.hh:166,265,370`). Two gaps remain:
1. `calculateFluxes_` truncates AD internally (`const Scalar trans = nbInfo.trans;` etc. at the
   top of the function) — a parameter-AD NBInfo would be cut off;
2. no lightweight way to build IntensiveQuantities-like objects with a caller-chosen Evaluation
   type (the model's IQs are TypeTag-fixed).

**Decision — keep the original fast assembly untouched for forward simulation; add a separate,
fully flexible cell evaluator used only by the adjoint, validated against the forward
linearizer via the replay archive** (the stored bitwise reference systems from Milestone A are
the strongest possible test harness for new linearizer code, and every flexible evaluation also
produces the plain value, which is checked against the production residual continuously):

- `opm/simulators/flow/adjoint/FlexibleCellEvaluator.hpp` (separate code, no production changes):
  - a **mini-IQ builder**: constructs `BlackOilFluidState<ParamEval,...>` + mobility/porosity
    from stored primary variables and problem tables with a caller-chosen
    `DenseAd::Evaluation<double, NP>`; fluid-system and material-law calls are already templated
    on the evaluation type, so this is composition of existing templated calls, not new physics
    (TPFA feature set only: no energy/polymer/solvent in v1);
  - a **generalized copy of `calculateFluxes_`** with deduced local types instead of `Scalar`
    locals (the only reason for code duplication; see upstreaming note below);
  - evaluation is **cell/face-local with known sparsity**, organized as **multiple passes with
    a small fixed set of AD types** rather than one pass with many slots. Exactly three
    instantiated evaluation types:
    - `Scalar` — *value pass*: recomputes the residual contribution with no derivatives; always
      run and **checked against the production/stored residual** (the "given value can be
      checked" principle — continuous self-validation of the flexible code path, per cell/face,
      not just at test time);
    - `DenseAd::Evaluation<double, numEq>` — *state-seeded validation pass*: reproduces the
      forward Jacobian blocks, compared against the replay archive (test level **T2p**,
      1e-12/bitwise target);
    - `DenseAd::Evaluation<double, 1>` — *parameter pass*: ONE parameter seeded per pass
      (pvmult of a cell, tmult of a face, one half-trans perm, one end-point scaler), repeated
      over the runtime-selected parameter list.
    Rationale for passes-over-slots: (a) gradient assembly runs **once per substep after the
    λ-solve**, not per Newton iteration — P passes ≈ 2P× one scalar assembly, negligible next
    to the forward solves; (b) a fixed slot count of 1 removes all slot-index bookkeeping and
    lets the parameter set be chosen at **runtime from the JSON config without recompiling**;
    (c) only three template instantiations keeps compile times down (a real constraint here).
    Multi-slot batching (e.g. `Evaluation<double,4>` chunks) is a later optimization, applied
    only if profiling shows the passes matter — the API (an evaluation context per pass) does
    not change.
  - parameter modes, in priority order: *poro/pvmult* (storage; cross-checks the analytic
    formula), *trans/tmult* (flux; analytic cross-check — flux confirmed linear in `trans` in
    the kernel, `transMult` separate), *perm* (T recomputed on the fly from half-trans with K
    seeded — needs `Transmissibility` to expose geometric half-trans factors), *end points*
    (seed the material-law scaling parameters in the mini-IQ's relperm call).

  **Precomputed-value provenance audit** (quantities computed *before* the linearization that
  hide parameter dependence and must be recomputed — or chain-ruled — inside the flexible
  evaluation; the parameter-side analog of the explicitly-updated-quantities catalog):

  | Precomputed quantity | Depends on | Treatment in flexible evaluator |
  |---|---|---|
  | `nbInfo.trans` (Transmissibility::update at init) | perm, MULT*, NTG | recompute in-loop from geometric half-trans × K for perm mode; scale directly for tmult mode |
  | reference porosity / pore volume (deck PORO×NTG×MULTPV) | poro, MULTPV | pvmult enters linearly at the PV level — analytic; decomposition needed only if differentiating PORO itself |
  | rock-compaction poro/trans multipliers | pressure history | state-dependent, handled by the state-seeded pass; no parameter dependence in v1 |
  | threshold pressures (THPRES) | deck or initial equil | constant w.r.t. our parameters when given in the deck; **if defaulted (computed from initial state) the dependence is ignored in v1 — warn** |
  | well connection factors CTF/WPI (Peaceman, Kh) | **perm** | NOT in the cell evaluator; dCTF/dK contributes to dJ/dperm through the well equations — **neglected in v1, documented + warned**, later added via the well-row machinery |
  | diffusivity/dispersivity factors | perm-like | out of scope v1 (isothermal, no diffusion — already a replay restriction) |
  | face areas, depths, gravity terms | geometry only | constant |
- **Well controls do NOT go through the cell evaluator**: dJ/du enters via the well-equation
  rows (derivative of the control equation w.r.t. the target — 2019 `duneCA_` design, or
  directly as rows of the #6751 full system). Unchanged from the existing plan.
- **Priority order** (per user): 1) poro/pvmult (analytic + AD cross-check), 2) trans/tmult
  (same), 3) perm (chain rule through half-trans first, on-the-fly AD second), 4) well controls.
- **Upstreaming path to kill the code copy later**: the only production patches needed for the
  flexible evaluator to share kernels with flow are (a) deduced local types in
  `calculateFluxes_` (mechanical, GPU-friendly), (b) optionally an IQ `update` templated on the
  evaluation type. Propose upstream once the separate evaluator is proven by T2p; until then the
  copy is tested bitwise against production every run.

Staging: **C.1** analytic pvmult/tmult assembler (unchanged Milestone C v1) → **C.2**
FlexibleCellEvaluator with state-seeded T2p validation → **C.3** parameter-seeded modes
(perm, end points) → **C.4** upstream type-generalization patches.

### C.2 scoping update (after Milestone C completion, 2026-06-12)

Code reading of `blackoilintensivequantities.hh` revises the
FlexibleCellEvaluator plan — **most of it already exists**:

- `BlackOilIntensiveQuantities::update(problem, priVars, globalSpaceIdx,
  timeIdx)` (line ~760) is the ElementContext-free path used by the TPFA
  assembly itself: public, decomposed into clean OPM_HOST_DEVICE steps
  (updateSaturations / updateRelpermAndPressures / updateRsRvRsw /
  updateMobilityAndInvB / updatePhaseDensities / updatePorosity), with
  static_asserts that exclude solvent/polymer/etc. — exactly the v1
  feature scope. Together with the public templated
  computeStorage/computeFlux kernels this IS the "standalone fully
  templated cell evaluation"; **no mini-IQ rewrite is needed**.
- **T2p (state-seeded validation)** is therefore implementable today
  with the live types: recompute IQ via the public update for both cells
  of a face, evaluate storage+flux, compare values AND state derivatives
  against the production linearization from the replay archive.
- **Parameter passes, revised:**
  - pvmult / tmult / perm:已 done analytically and FD-verified — the
    evaluator is NOT needed for them.
  - **End-point scaling (the remaining parameter class): cell-local
    parameter FD** is the right v1 (as sketched in the original plan):
    perturb the cell's scaled saturation end points in the
    materialLawManager, recompute IQ(cell) + storage + the cell's face
    fluxes through the public kernels, difference → dR/dθ columns,
    contract with λ. Cost: O(2 × params/cell) cell-local evaluations per
    substep — no global solves. Needs one more adjoint-hooks accessor
    (mutate per-cell EclEpsScalingPoints in EclMaterialLawManager).
  - **Exact parameter-AD** for end points would require the material-law
    *parameter* objects to be templated on the evaluation type (they are
    Scalar-based EclEpsScalingPoints today) — an upstream-sized change;
    long-term C.4, not on the critical path.

### Extension architecture: inheritance / separate module vs in-tree edits

Question: can the adjoint live mostly in derived classes (or a separate repo, like
opm-flowgeomechanics) instead of modifying opm-simulators, to minimize merge conflicts?

**Verified extension surfaces:**
- `FlowProblem` lifecycle methods are `virtual`: `beginEpisode`, `beginTimeStep`,
  `endTimeStep`, `endEpisode`, `initialSolutionApplied` (`FlowProblem.hpp:307-1001`) — a
  derived problem can host the recorder with zero core changes. This is exactly the geomech
  pattern: `EclProblemGeoMech : public FlowProblemBlackoil<TypeTag>` with
  `beginTimeStep()/endTimeStep() override` calling `Parent::...`, installed via a property
  override `struct Problem<TypeTag, TTag::EclFlowProblemMech>` in the module's own TypeTag,
  own executables, separate repo with `opm_need_version_of("opm-simulators")`.
- The property system is the designed downstream-override mechanism (Problem, WellModel,
  Linearizer are all replaceable per TypeTag); a derived `BlackoilWellModel` can expose
  protected members (e.g. `updateNupcolWGState`).
- The replay/adjoint driver only calls **public** simulator APIs (verified during Milestone A)
  — it works unchanged from any module.
- NOT extensible by inheritance: `SimulatorFullyImplicit` (no virtuals — downstream modules
  simply use their own driver/main instead) and the `AdaptiveTimeStepping` substep loop (no
  virtual hook — the reason the accepted-substep callback edit was needed; a derived-Problem
  `endTimeStep` override is the inheritance-equivalent hook at the same point).

**Current conflict surface (measured):** the entire adjoint branch changes production files by
~106 lines in 5 files (AdaptiveTimeStepping hook ~28, SimulatorFullyImplicit wiring ~65,
BlackoilWellModelGeneric helper ~13) + cmake lists; all 2k remaining lines are new files under
`flow/adjoint/`, `linalg/transposeMatrix.hpp`, `tests/`.

**Decision — hybrid discipline:**
1. **Stay in-tree for now** on the `adjoint` branch: the conflict surface is already minimal
   and rebases trivial; it preserves the single-binary `flow_blackoil --adjoint-mode=replay`
   workflow (a separate repo necessarily means its own big-TU binary).
2. **Enforce the downstream-ready rule**: new functionality only as new files under
   `flow/adjoint/`; production edits only as small *generic* hooks that stand alone as upstream
   micro-PRs (the substep callback mirrors the existing tuningUpdater pattern and is useful
   beyond the adjoint; the WGState normalization is generally useful for pre-step snapshots).
3. **Move to a separate `opm-flowadjoint` module (geomech-style) when** non-upstreamable
   material accumulates (optimization loops, objective/config tooling) or upstream PR friction
   appears. The move is cheap because of rule 2: the recorder switches from the callback to a
   derived-Problem `endTimeStep`/`beginEpisode` override, the replay driver and storage move
   unchanged, and the module brings its own TypeTag + binary.

### Milestone C — parameter gradients (v1: dJ/dPV(φ) and dJ/dT_face)

Accumulated during the backward sweep, no extra solves:
- **Porosity/PV:** storage linear in φ ⇒ `g_φ[i] += λ_k[i]·(V/dt)(S_i(x_k) − S_i(x_{k-1}))/φ_i`,
  using the storage values the replay already computes.
- **Transmissibility:** TPFA flux linear in T per face ⇒ face loop over `neighborInfo_` at converged
  IQs, `g_T[f] += (λ_k[I] − λ_k[J])·flux_f/T_f`. v1: isothermal, no diffusion/dispersion.
- **Perm chain rule** (follow-up PR): post-process g_T through half-transmissibilities in
  `opm/simulators/flow/Transmissibility.hpp`.
- **End-point scaling** (sketch): cell-local parameter FD on storage+flux contributions first;
  AD on material-law params later.

**Verification:** central finite differences perturbing MULTPV/PERMX cell-wise on MODEL_1D_DEBUG
(all cells) and a few SPE1 cells, rel-tol ~1e-4; comparison with `adjoint_results` references from
`add_adjoint_tests`. Extra channel: JutulDarcy reads OPM decks — run the same deck + objective
through `solve_adjoint_sensitivities` in JutulDarcy and diff gradients (script, not CI).

### Serialization review (requirement 4)

Chosen: HDF5 via existing `HDF5Serializer` + `serializeOp`/MemPacker — battle-tested by
save/load-step, versioned header, single browsable file. Rejected: 2019 boost+ewoms-ascii (two
formats, no schema), MatrixMarket sidecars (kept only as optional debug via existing
`TpfaLinearizer::exportSystem`), bespoke binary. Results (λ, gradients, J) written into the same
file; observed well curves read via ESmry — no new input format.

### PR sequencing

0. **PR0 (prerequisites, already open)**: land [#7039](https://github.com/OPM/opm-simulators/pull/7039)
   (+ opm-common `fix_replay`) — gives T0; push [#6751](https://github.com/OPM/opm-simulators/pull/6751)
   toward merge in parallel (needed by PR4, not before).
1. **PR1** recorder + parameters + IO + AdaptiveTimeStepping callback (off by default) ⇒ T1
2. **PR2** transposeMatrix + AdjointLinearSolver (UMFPack + cprt-via-FlexibleSolver) + unit tests
3. **PR3** `flow/flow_adjoint.cpp` binary + AdjointReplay + replay regression test ⇒ T2+T3,
   **Milestone A done**
4. **PR4** Bdiag pass + WellCurveObjective (JSON config + ESmry) + AdjointSolver on the #6751
   full system + φ/T multiplier gradients + sign/FD tests ⇒ T4+T5, **B+C v1**; JutulDarcy
   cross-validation script alongside (T6)
5. **PR5** perm chain rule, well-storage cross term, matrix-free transposed operator (if needed),
   end-point scaling, dJ/du well-control gradients, MPI

### Risks / early spikes

1. Well re-assembly fidelity at the converged point (control switching, THP/VFP, groups) — week-1
   spike: replay one SPE1 step, diff B/C/D/resWell bitwise; fallback `assembleWellEqWithoutIteration`.
   Restrict v1 to simple BHP/rate controls, no networks.
2. Sign conventions (residual sign, extract()'s subtraction, objective direction) — pin with a
   1-cell/1-well/2-step FD test first.
3. `needsTimestepInit`/`markTimestepInitialized` choreography in replay.
4. Aquifer/tracer/drift cross-step terms: replayed exactly (A fine) but ignored in adjoint —
   exclude from gradient tests; `--enable-drift-compensation=false` for gradient runs.
5. DRSDT/VAPPARS (non-recycled storage cache) — add a dedicated replay test before claiming support.
6. Serial-first throughout; MPI explicitly deferred (UMFPack, single-rank replay).
7. Dependency risk: Milestone B's primary path assumes [#6751](https://github.com/OPM/opm-simulators/pull/6751)
   merges; the Schur-transpose fallback is specified above precisely so PR4 isn't blocked on it.
   Milestone A assumes [#7039](https://github.com/OPM/opm-simulators/pull/7039) semantics for
   failed-update resetting — rebase the adjoint branches on it from day one.
8. cprt has likely not been exercised in years — the v1.5 spike (cprt vs ILU0 on the transposed
   system) doubles as a health check; UMFPack v1 means nothing blocks on it.
