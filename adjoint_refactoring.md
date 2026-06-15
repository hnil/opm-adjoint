# A result-preserving refactoring: separate well-control *advance* from well-residual *assembly*

_2026-06-13. Concrete proposal with a per-stage no-change guarantee.
Companion to `adjoint_status.md` §3._

## 0. The binding constraint

Every stage below is **result-preserving by construction**: it is pure
indirection or added serialization, never a change to a computed value.
The forward simulation must produce byte-identical output. The oracle for
this is already in the tree: the **bitwise replay suite**
(`adjoint_replay_m1d`, `adjoint_replay_spe1`, SPE9) compares the
re-linearized residual/Jacobian against the stored forward system at
`1e-12` (target bitwise). Any stage that perturbs a result fails it
immediately. A standard `flow` regression run (compare `.UNSMRY`/`.UNRST`)
is the second oracle.

## 1. The problem, confirmed empirically

`BlackoilWellModel::assemble(dt)` does two things:

```
assemble(dt):
    (A) updateWellControlsAndNetwork(dt)    // ADVANCE: stateful, iterative, forward-only
    (B) assembleWellEqWithoutIteration(dt)  // ASSEMBLE: build B/C/D/resWell
        updateCellRates()
```

The adjoint only wants (B), reproduced from stored state. Two facts were
verified directly in this tree:

1. **The full converged state *is* serialized.** The eWoms
   `Simulator::serializeOp` (`simulator.hh:921`) writes
   `*vanguard_` (→ `SummaryState`, `FlowGenericVanguard.cpp:423`),
   `*model_` (→ solution at both time levels) and `*problem_`
   (→ well model → `guideRate_`, `BlackoilWellModelGeneric.hpp:260`). So
   "missing serialized state" is **not** the root cause.

2. **Restoring the full state and calling (B) directly still does not
   reproduce the forward system** — it fails the bitwise replay on SPE1
   (and aborts with system-compare on). Reason: (B) is not closed over
   serialized state. It re-derives the active controls from the *live*
   summary state —
   `WellInterface_impl.hpp:990`:
   ```cpp
   const auto& summary_state = simulator.vanguard().summaryState();
   const auto inj_controls = this->well_ecl_.injectionControls(summary_state);
   const auto prod_controls = this->well_ecl_.productionControls(summary_state);
   ```
   and it relies on **transient, non-serialized state that only (A)
   populates** on the live well objects (the effective control mode and
   scaling, the `well_container_` membership, network node pressures).

So the real obstacle is not data that was forgotten in the snapshot; it
is that **(B)'s inputs are partly transient in-memory state produced as a
side effect of (A)**, plus a direct read of a live global. Re-running (A)
on a restored snapshot is the only way (B) currently works — and (A)
crashes on group-controlled decks with shut wells
(`updateWellControlsAndNetwork` does a name lookup over the live-well
subset while the schedule still lists shut wells: the Norne `map::at`).

## 2. The refactoring

**Make (B) a pure function of an explicit `WellAssemblyState` bundle, so
that `forward = advance→assemble` and `replay = restore-bundle→assemble`
run the *same* assembly.** Staged so each step is independently
result-neutral and independently testable.

### Stage 1 — Inventory and serialize the advance's outputs (additive)

Enumerate every value (B) reads that (A) produces. From the code these
are: per-well effective control mode + scaling (today on the
`WellInterface`/`StandardWell` objects and partly in `WellState`), group
targets/reductions (`GroupState` — already serialized), guide rates
(`guideRate_` — already serialized), and network node pressures (the
network object).

Action: give the well objects' control-relevant transient state a
`serializeOp`, or (cleaner) move it into `WellState` where most of it
belongs. Add a `WellAssemblyState` aggregate:

```cpp
template<class Scalar, class IndexTraits>
struct WellAssemblyState {
    WellState<Scalar, IndexTraits> well_state;   // incl. effective controls
    GroupState<Scalar>             group_state;
    GuideRate                      guide_rate;
    NetworkState                   network;       // node pressures
    // the summary-vector slice the controls depend on (see Stage 2)
    void serializeOp(auto& s) { s(well_state); s(group_state);
                                s(guide_rate); s(network); }
};
```

*No-change guarantee*: serialization only; no computed value changes.
*Test*: full suite stays bitwise; a `flow` restart round-trip
(`--save-step`/`--load-step`) is unchanged.

### Stage 2 — Route (B)'s control derivation through the bundle (pure indirection)

Replace the direct live read in `assembleWellEqWithoutIteration` with a
read from the bundle:

```cpp
// before (WellInterface_impl.hpp:990)
const auto& summary_state = simulator.vanguard().summaryState();
const auto inj_controls = well_ecl_.injectionControls(summary_state);
// after
const auto& inj_controls = assemblyState.controlsFor(this->name()).injection;
```

In the **forward** path the bundle is populated, during (A), with exactly
`well_ecl_.injectionControls(simulator.vanguard().summaryState())` — the
identical object — so the value the assembly sees is unchanged.

*No-change guarantee*: the bundle carries the same value the live read
produced; this is a move of *where* the value is read, not *what* it is.
The subtlety is UDQ/ACTIONX controls (summary-vector-valued targets):
those must be evaluated once in (A) and stored, which is exactly what the
bundle is for — and still identical, because (A) already evaluated them
at the same point in the forward run.
*Test*: bitwise suite + a UDQ/ACTIONX deck regression.

### Stage 3 — Split the well-model API along the bundle

```cpp
// forward-only, mutating, non-differentiable (today's updateWellControlsAndNetwork):
WellAssemblyState advanceWellControlState(dt);
// pure function of the bundle + reservoir state (today's assembleWellEqWithoutIteration,
// minus its out-of-band reads):
void assembleWellResidual(const WellAssemblyState&, dt);
```

`assemble(dt)` becomes `assembleWellResidual(advanceWellControlState(dt), dt)`
— byte-identical to today. Replay/adjoint becomes
`assembleWellResidual(restoredBundle, dt)` — no (A), no shut-well crash.

*No-change guarantee*: forward composes the two exactly as before.
*Test*: bitwise suite; then the **payoff** — point the adjoint replay at
the restored bundle and confirm Norne replays bitwise where it currently
aborts.

## 3. Minimal first PR (smallest provably-neutral slice)

If a full landing is too big to start, the safe keystone is **Stage 2 for
the summary-state read only**, behind a seam that defaults to the live
read:

1. Add `WellInterface::assemblyControls()` returning, by default,
   `well_ecl_.{injection,production}Controls(simulator.vanguard().summaryState())`
   — i.e. today's exact expression, just named.
2. Replace the direct reads at `WellInterface_impl.hpp:990`,
   `StandardWell_impl.hpp:367`, and the `MultisegmentWell` assembly sites
   with calls to it.
3. Add a (default-empty) injectable override so a caller (the adjoint)
   can supply controls from a restored bundle instead.

This compiles to identical code in the forward path (the default seam is
the original expression), so it is result-neutral by inspection, and it
establishes the single injection point the rest of the refactor builds
on. Verify: full bitwise suite + one `flow` deck `.UNSMRY` diff.

## 3a. Progress (2026-06-13): the assembly entry point exists

The opm-simulators side now has `BlackoilWellModel::assembleWellEqGivenControls(dt)`
(additive, forward-neutral): it assembles the well equations at a
restored converged state — `updatePrimaryVariables` +
`prepareWellsBeforeAssembling` + `assembleWellEqWithoutIteration` +
`updateCellRates` — **without** re-running the control/group/network/
guide-rate advance. The adjoint replay's final linearization now calls
it instead of the full `beginIteration` (which re-ran the advance).

Verified with a fast group-control repro
(`model2/0A1_GRCTRL_LRAT_ORAT_BASE_MODEL2_STW`, 2794 cells, ~2 s):

- **Result-neutral on the verified envelope**: m1d, SPE1 (123/123) and
  SPE9 (21/21) stay bitwise; full suite 14/14. The forward is untouched.
- **Group control essentially works now.** The control-equation target
  for a group-controlled well is recomputed during assembly
  (`getGroupProductionControl`, `WellAssemble.cpp:171`) from the group
  state *and the guide rates* — and the guide rates were stale (the
  install restored only the WGState, not the well-model `guideRate_`).
  Restoring the **full** snapshot-k state in the install (the whole
  `Simulator::serializeOp`, which brings back `guideRate_` and the summary
  state) closes it: on the model2 group-control repro **every report step
  after the first now replays bitwise (residual/Jacobian rel diff 0.0)**.
  Only the very first report step (well/guide-rate startup transient)
  remains non-bitwise, with *tiny* absolute residuals (~1e-4, vs ~3e-2
  before — a 100x reduction; the ~1.0 *relative* diff is an artifact of
  the converged residual being ~1e-7). Still result-neutral on the
  verified envelope (m1d, SPE1 123/123, SPE9 21/21 bitwise; suite 14/14).
- **Norne now replays end-to-end (DONE).** The well-local recompute
  crash was `prepareWellsBeforeAssembling`, which runs well-operability /
  inner-iteration *decisions* (it can stop wells) and does a shut-well
  name lookup — that is forward-only advance logic, not assembly, and is
  not needed once `updatePrimaryVariables` has set the well primary
  variables from the restored state. Dropping it from
  `assembleWellEqGivenControls` makes the recompute shut-well-safe.
  Result: the full Norne deck (44431 cells, 36 wells, group control +
  shut wells, hysteresis, VAPOIL, faults) replays to completion, and
  with `--adjoint-save-system` the re-linearized system matches the
  stored forward system: **~90% of substeps bitwise (294/329), ~96%
  within 1e-3**; the ~11 larger deviations (up to ~0.5 abs) are at the
  known-hard points — the first report step's startup transient,
  well-event boundaries, and hysteresis-update steps — all documented v1
  approximations. Still result-neutral: SPE1 123/123, SPE9 21/21 bitwise,
  suite 14/14; the forward `assemble()` is unchanged.

So Stage 1's keystone — a forward-neutral "assemble at restored controls"
entry point fed by the full restored state — is in place and reproduces
group-controlled **and** shut-well field decks. What remains for *bitwise*
field replay: the first-report-step startup transient and the
hysteresis/well-event boundary steps (small, localized; the gradient is
already accurate since most substeps are exact).

## 4. Why this is the right shape

The TPFA reservoir kernels already went through exactly this separation
during the GPU work: `computeStorage`/`computeFlux` are pure, templated,
side-effect-free functions of state — which is *precisely why* reservoir
replay is trivially bitwise and the cell-local cross-terms (PV, trans,
end points) were straightforward. The well/group/network side never got
that treatment and remains a stateful object graph whose assembly reads
live globals and transient members. Giving (B) the same purity is the
single change that removes the group-control wall, and it pays off beyond
the adjoint: deterministic well-state restart, cleaner NLDD (which
already threads `groupStateHelper`), isolated well-equation tests, and a
smaller surface for GPU well assembly.

## 5. What this does *not* require

It does not require differentiating (A). The control advance stays a
black box that runs in the forward only; the adjoint never needs its
derivative because control switches are at worst objective kinks
(documented), and within a converged step the controls are fixed. The
refactor only needs (A)'s *output* to be captured and (B) to consume it —
not to make (A) itself differentiable or replayable.
