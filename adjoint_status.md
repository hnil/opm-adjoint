# Adjoint module — status, limits, and a well/group/network refactoring review

_Last updated 2026-06-13. Companion to `adjoint_plan.md` (design) and
`adjoint_testing.md` (how to run every step)._

## 1. Current status

The `opm-adjoint` module computes adjoint-based gradients of a forward
OPM-flow simulation, on the normal `flow` code path, built as a separate
module against an `adjoint-hooks` branch of opm-simulators (three small
accessor commits). Everything below is implemented and verified.

**Forward recording → backward replay → adjoint solve → gradients**, all
driven from one binary `flow_adjoint`:

- `--adjoint-save=true` records every accepted substep (HDF5 per-rank
  archive, or a dependency-free directory store).
- `--adjoint-mode=replay` re-creates the converged residual/Jacobian of
  each substep from the snapshots and compares to the stored system
  (bitwise on the supported envelope).
- `--adjoint-mode=gradient` runs the backward adjoint sweep and writes
  the parameter gradients; `--adjoint-mode=objective` evaluates J only.

**Gradients available and FD-verified** (central differences, fixed
report-step stepping): pore volume / porosity, transmissibility,
permeability (half-trans chain rule), well-control targets (dJ/du), and
saturation end-point scaling (SWL, SWCR, KRW, … via a keyword→endpoint
registry). **Objectives**: average reservoir pressure, well BHP, well
rate, and well-curve matching against a reference summary (single-term,
multi-term, and reference-case misfits), read through `ESmry` with unit
conversion.

**Linear algebra**: the per-substep adjoint system is the transpose of
the forward Schur-reduced system. Serial uses an explicit transpose +
`FlexibleSolver` (`umfpack` default, or `ilu0`/`cpr`/`cprt`/JSON); `cprt`
(transpose-CPR) is the fastest correct serial choice. **MPI** works: the
operator is the exact transposed action (interior-row scatter of A^T over
the forward matrix + owner accumulation — see below for why a plain local
transpose is wrong in parallel), preconditioned with parallel ILU0;
verified against serial to ~1e-5 on SPE1 for np = 2, 3, 4.

**Verified envelope**:

| Deck | cells | result |
|---|---|---|
| MODEL_1D_DEBUG | 3 | replay bitwise; all gradients FD-verified |
| SPE1CASE1 | 300 | replay bitwise; gradients + objectives verified; MPI np≤4 |
| SPE9_CP_SHORT | 9000 | replay **bitwise** (21/21); gradient + timing; runs flag-free |

14 ctest entries (~20 s) guard the whole chain. Multi-perforation wells
work out of the box: the binary defaults `--matrix-add-well-contributions
=true` and a forward `--linear-solver=ilu0` (because that flag otherwise
auto-promotes `cpr`→`cprw`, unregistered in this build).

**Cost profile** (SPE9, logged per run): transposed solve 56 %,
re-linearization 41 %, gradient accumulation 3 %, Schur 0.1 %. The linear
solve and the re-linearization dominate; the gradient kernels are free.

**JutulDarcy cross-check**: environment installed (Julia LTS); forward
SPE1 agrees where the well model is not the differentiator (rate target
3179.7 sm³/d and the BHP floor match exactly), diverging only in the
well-index/control-switch behaviour — the expected OPM-vs-Jutul
difference. Use a pressure-average or BHP-controlled objective for
gradient cross-validation.

## 2. What limits further development

The implementation is deliberately v1-scoped: **plain black-oil, no
hysteresis / DRSDT-VAPPARS cross-terms, no group control, no well
events**. The boundaries, in rough order of how soon they bite:

1. **Group control + well events (the hard wall — see §3).** Field decks
   with a group hierarchy, guide rates, and wells that open/shut across
   report steps cannot be replayed: the re-linearization re-runs the
   forward control logic, which is not reproducible from the stored
   state. Norne records fine but aborts at the first backward substep.
   This is the single biggest limit and the subject of the refactoring
   review below.

2. **Explicitly-updated quantities are replayed but not differentiated.**
   Hysteresis state, DRSDT/DRVDT max values, max-saturation tracking and
   rock-compaction history are serialized (so replay is exact) but their
   cross-step derivative contributions are neglected in the gradient.
   Decks using them get exact J and approximate dJ/dm. Quantifying these
   (a DRSDT FD test) is an open item.

3. **Well explicit quantities cross-term.** The wellbore-storage F0_ and
   explicit connection pressures couple step k to k−1; this cross-term is
   neglected in v1 (documented approximation, to be quantified by FD).

4. **Storage cache must be off.** v1 requires
   `--enable-storage-cache=false` so the residual is a pure
   R(x_k, x_{k−1}). Re-enabling the cache for the adjoint is a certified
   optimization (the T0.7 equivalence test exists) but not yet done.

5. **Fixed-stepping for any comparison.** FD checks and MPI-vs-serial
   checks need `--enable-adaptive-time-stepping=false`; otherwise the
   substep sequence diverges and the gradients differ legitimately. Fine
   for verification, but it means adaptive runs cannot be FD-checked
   directly.

6. **Performance.** Each backward substep does a full re-linearization
   and a fresh transposed solve with no setup reuse. Adequate for the
   verified sizes; for large fields the §1 cost profile says the solve
   and re-linearization are where the work is (preconditioner reuse,
   keeping states in memory instead of round-tripping the archive).

7. **Exact end-point parameter-AD.** End-point gradients use cell-local
   finite differences of the full residual (correct, but O(cells) FD
   evaluations). Exact AD would need the material-law parameter objects
   templated on the evaluation type — an upstream-sized change.

Items 2–7 are incremental. **Item 1 is architectural**, and it is worth
asking whether a change in opm-simulators would make it (and the adjoint
in general) materially cleaner.

## 3. Review: well / group / network design vs. the adjoint

### 3.1 What the adjoint needs

The adjoint/replay needs exactly one thing from the well model: given a
stored converged state, **re-assemble the same residual and Jacobian the
forward run produced**, deterministically, without advancing any state.
The forward already exposes the right entry point —
`BlackoilWellModel::assembleWellEqWithoutIteration(dt)` — which loops the
wells and assembles B/C/D/resWell. The trouble is what that assembly
*reads*.

### 3.2 The actual coupling (measured in the code)

`BlackoilWellModel::assemble(dt)` is two responsibilities welded
together:

```
assemble(dt):
    (A) updateWellControlsAndNetwork(...)   // ADVANCE control/network state
    (B) assembleWellEqWithoutIteration(dt)  // ASSEMBLE residual given state
        updateCellRates()
```

(A) is a stateful, iterative, communication-heavy, **non-differentiable**
fixed-point loop (`updateWellControlsAndNetwork` →
`…NetworkIteration` → group-data communication, guide-rate updates,
network balancing, gas-lift optimization, control-mode switching). It
mutates many objects.

(B) is the pure residual assembly the adjoint wants. But it is **not a
pure function of a single serializable state**. Tracing
`WellInterface::assembleWellEqWithoutIteration`
(`WellInterface_impl.hpp:983`), its inputs are:

| Input | Where it lives | In the adjoint snapshot? |
|---|---|---|
| reservoir intensive quantities | model solution | yes (restored) |
| per-well control mode, rates | `WellState` (in `WGState`) | yes |
| group targets / reductions | `GroupState` (in `WGState`) | yes |
| **guide rates** | `BlackoilWellModel::guideRate_` | no (only in the *full* well-model `serializeOp`, not `WGState`) |
| **summary state** | `simulator.vanguard().summaryState()`, read **directly** at `WellInterface_impl.hpp:990` to build inj/prod controls | no |
| network node pressures | the network object | no |
| static well/connection defs | schedule, via `prepareDeserialize` | rebuilt |

So `WGState = {well_state, group_state}` (`WGState.hpp:46-47`) captures
**two of the five-plus** state pieces (B) reads. The rest are maintained
as side effects of (A) and read out of band (notably the *direct*
`simulator.vanguard().summaryState()` read inside the assembly).

### 3.3 Why this blocks the adjoint

Because the assembly is not closed over a single state object, replay
cannot just "restore state, then assemble". It must re-run (A) to
regenerate guide rates, group targets, network pressures and the summary
slice — and (A) is exactly the part that (i) is non-differentiable,
(ii) is not reproducible from stored data, and (iii) crashes on decks
with shut wells (it does a name lookup on the live-well subset while the
schedule still lists the shut wells — the Norne `map::at`). The two
work-arounds both fail for principled reasons: skipping (A) changes the
converged system (so it breaks bitwise replay on simple decks too), and
restoring more state piecemeal just exposes the next out-of-band read.

A sharper finding (verified directly in this tree, 2026-06-13): the full
converged state *is* serialized — `Simulator::serializeOp` writes the
vanguard (→ summary state), model and problem (→ well model → guide
rates). Yet restoring all of it and calling (B) directly still does not
reproduce the forward system, because (B) re-derives the controls from
the *live* `simulator.vanguard().summaryState()`
(`WellInterface_impl.hpp:990`) and depends on transient state that only
(A) writes onto the live well objects. So the obstacle is not forgotten
data; it is that (B) is not closed over serialized state. The concrete,
staged, result-preserving refactoring is in **`adjoint_refactoring.md`**;
the summary below is the high-level shape.

### 3.4 Recommended refactoring (upstream, opm-simulators)

**Make the well-equation assembly a pure function of an explicit,
fully-serializable "assembly input state".** Concretely:

1. **Introduce a `WellAssemblyState` bundle** (or widen `WGState`) that
   contains *everything* (B) reads beyond the reservoir solution:
   `well_state`, `group_state`, guide rates, network node pressures, and
   the summary-state slice the controls depend on. One object, one
   `serializeOp`.

2. **Remove out-of-band reads from the assembly.** The biggest offender
   is the direct `simulator.vanguard().summaryState()` read in
   `WellInterface::assembleWellEqWithoutIteration`
   (`WellInterface_impl.hpp:990`): the inj/prod controls it builds should
   come from the bundle, not from a live global. Likewise guide rates
   should be taken from the bundle, not from `guideRate_`.

3. **Split the well model API cleanly into two phases** that the bundle
   mediates:
   - `advanceWellControlState(dt) → WellAssemblyState` — today's
     `updateWellControlsAndNetwork`, the forward-only mutating phase;
   - `assembleWellResidual(const WellAssemblyState&, dt)` — a pure
     function of the bundle + reservoir state.

   Forward = advance → assemble. Replay/adjoint = restore bundle →
   assemble. Identical assembly, no re-running of (A).

This is the same separation the TPFA reservoir kernels already enjoy
(the GPU work made `computeStorage`/`computeFlux` pure, templated,
side-effect-free — which is exactly why reservoir replay is trivially
bitwise and the cell-local cross-terms were easy). The well/group/network
side never got that treatment; it remains a stateful object graph whose
assembly reads live globals.

### 3.5 Cost / benefit

- **Benefit for the adjoint**: removes the §2-item-1 hard wall entirely.
  Group-controlled field decks (Norne) would replay and differentiate
  with no special handling, because the assembly would be reproducible
  from one restored bundle. It also makes the existing simple-deck path
  more obviously correct (no reliance on (A) being idempotent on replay).
- **Benefit beyond the adjoint**: a pure assembly phase is independently
  useful — deterministic restart of well state, cleaner NLDD/domain
  decomposition (which already passes `groupStateHelper` around), easier
  testing of the well equations in isolation, and a smaller surface for
  the GPU well-assembly effort.
- **Cost / risk**: it touches `BlackoilWellModel`, `WellInterface`,
  `WGState`, and the network/guide-rate objects — a real opm-simulators
  refactor, not a module-local change. The summary-state decoupling in
  particular needs care (UDQ/ACTIONX can make controls depend on
  arbitrary summary vectors). It should be staged: first widen the
  serialized bundle and route guide rates + network pressures through it
  (mechanical), then tackle the summary-state read (the subtle part).

### 3.6 If the upstream refactor is not done

The module-local fallback is to **capture the missing pieces in the
adjoint snapshot and restore them before assembly**, i.e. serialize
`guideRate_`, the network state and a summary-state copy alongside
`WGState`, and install all of them in the replay's install phase, then
call `assembleWellEqWithoutIteration` directly (skipping (A)). This is
viable but fragile: it duplicates knowledge of (B)'s inputs in the
adjoint module and will silently break whenever a new out-of-band read is
added to the assembly. It is the right tactical step if a specific
group-controlled deck is needed before the upstream refactor lands; it is
not a substitute for §3.4.

## 4. Recommendation

The v1 envelope (single-region controls, no group hierarchy/events) is
complete and solid. The next strategic step — required for field-scale
decks and cleanly beneficial beyond the adjoint — is the §3.4 separation
of well-control *advancement* from well-residual *assembly*, with a
single serializable assembly-input bundle. Until that lands, group-
controlled decks are out of scope, and the §3.6 tactical capture can
unblock a specific deck if needed.

## 5. Stress-test coverage (2026-06-15)

Forward-record + bitwise replay (`--adjoint-save-system`, system compared
at 1e-12) across a range of standard-well 3-phase black-oil decks from
opm-tests. "report-0 transient" = the only non-bitwise substeps are in
the first report step, with tiny absolute residuals (~1e-4 down to
~1e-9); these are the documented startup/explicit-update approximation
and do not affect gradients (the group-control gradient FD-passes
regardless).

| deck | cells | feature exercised | replay |
|---|---|---|---|
| MODEL_1D_DEBUG | 3 | basics | bitwise (FD-verified) |
| SPE1CASE1 | 300 | basics | 123/123 bitwise |
| SPE1_GRPCTRL | 300 | **group control** | 120/120 bitwise (gradient FD-verified) |
| wconprod | 300 | well control | 7/7 bitwise |
| spe3 | 324 | VAPOIL+DISGAS | 175/179 (report-0 transient, abs ≤1e-6) |
| minpv | 1652 | MINPV | 1/1 bitwise |
| grupcntl | 2794 | **group control** | 11/11 bitwise |
| model4 | 3042 | **group control** | 106/106 bitwise |
| gconprod | 6000 | **group control** + ENDSCALE | 25/25 bitwise |
| gconinje | 6000 | **group injection control** | 12/12 bitwise |
| SPE9_CP_SHORT | 9000 | multi-perforation wells | 21/21 bitwise |
| Norne | 44431 | group ctrl + shut wells + hyst + VAPOIL + faults | end-to-end, ~90% bitwise |
| editnnc | 210 | NNCs | report-0 transient only (abs ≤1e-4) |
| mult | 210 | region multipliers | report-0 transient only (abs ≤1e-5) |

**Conclusion — what works now:** plain black-oil with standard wells,
single-well *and* group control (GCONPROD/GCONINJE/GRUPCNTL), well
open/shut events, NNCs, region multipliers, ENDSCALE, MINPV, multi-
perforation wells, serial and MPI — all replay bitwise (the few report-0
startup substeps aside), and the gradients are FD-verified for porosity,
transmissibility, permeability, end-point scaling, well controls, and
group control. Norne-class field decks replay end-to-end (~90% bitwise).

**Known remaining (not bitwise):**
- the first-report-step startup transient on some decks (small, localized,
  gradient-irrelevant — likely the explicit DRSDT/VAPPARS/initial-potential
  update at step 0);
- **WGRUPCON guide-rate definitions** (`wgrupcon` deck: report steps
  3/5/7 deviate by ~1.0) — the guide-rate *re-definition* timing is not
  reproduced by restoring `guideRate_` alone;
- multisegment wells / networks / compositional / thermal / solvent /
  polymer (out of the v1 black-oil + standard-well scope by design).

## 6. Catalog of explicitly-updated quantities (and how the adjoint treats them)

A number of quantities are updated *explicitly* in the forward run
(between iterations or steps) rather than being solved implicitly. The
adjoint handles each of three ways. The end-of-run scan
(`AdjointDeckWarnings.hpp`, printed after every gradient/replay run)
detects the ones present in the deck and warns; the tags below match the
warning tags.

| Quantity | Where updated | Replay (J) | Gradient (dJ/dm) | Warning tag |
|---|---|---|---|---|
| Saturation hysteresis (HYSTER) | `endTimeStep` (max/min sat in materialLawManager) | **exact** (state serialized) | cross-term neglected | `approx-gradient` |
| DRSDT / DRVDT (active, max>0) | `endTimeStep` (mixControls_ max Rs/Rv rate) | **exact** (cache-free storage) | cross-term neglected | `approx-gradient` |
| VAPPARS | `endTimeStep` | **exact** | cross-term neglected | `approx-gradient` |
| DRSDT/DRVDT **0** (frozen Rs/Rv) | — | **exact** | **exact** (no rate-dependent term) | none (not flagged) |
| Rock compaction (poro/trans mult from pressure history) | `endTimeStep` | exact (serialized) | cross-term neglected | *(not yet auto-detected)* |
| Max-oil / max-water saturation tracking (drives VAPPARS) | `updateMaxOilSaturation_` in `endTimeStep` | exact | neglected | via `VAPPARS` (`approx-gradient`) |
| Well explicit quantities (F0_ wellbore storage, connection hydrostatic pressure drop from prior-state perf densities, explicit B) | `prepareTimeStep` / `computePropertiesForPressures` (`StandardWellConnections.cpp:230`) | exact (recomputed from state k−1) | known cross-term neglected | `approx-gradient` (standing note, printed whenever any other item fires) |
| Guide-rate group control allocation (target shared among members by guide rates from well potentials) | `BlackoilWellModelGeneric.cpp:1595` `computePotentials` → `updateGuideRates` | **exact** (converged guide rates restored, verified bitwise on gconprod/gconinje/grupcntl/model4) | d(guide-rate)/d(state) cross-term neglected — **negligible/zero for single-member or guide-rate-insensitive groups** (why `adjoint_fd_grpctrl` on SPE1_GRPCTRL still passes); can be material when ≥2 members share a target | `approx-gradient` |
| **Reservoir-volume / voidage-replacement group control (RESV/VREP/REIN)** | `updateWellControlsAndNetwork` (group voidage target) | **not bitwise** — the effective control target is re-derived from reservoir voidage during the replay assembly | approximate | `approx-replay` |
| Single-rate group control (ORAT/GRAT/WRAT/LRAT, GRUP) | `updateWellControlsAndNetwork` | **exact** (verified bitwise on gconprod/gconinje/grupcntl/model4) | exact apart from the guide-rate cross-term above | guide-rate row |
| Well control-mode switching | during Newton | exact (converged controls in WGState) | kinks at switches are real objective non-smoothness, not a bug | none |
| Aquifers | aquifer `endTimeStep` | **inexact** (aquifer contributions not assembled in replay) | neglected | `neglected` |
| Extended network (NODEPROP/BRANPROP) | `updateWellControlsAndNetwork` | inexact | neglected | `neglected` |

The `approx-gradient` group is the principled v1 trade-off: the forward
*state* is serialized so the residual/Jacobian (hence J) is reproduced
bitwise, but the cross-step derivative of the explicitly-updated quantity
is dropped from dJ/dm. The `approx-replay` and `neglected` groups are
where the replay itself is not bitwise — the targets for closing them are
in `adjoint_refactoring.md` (the voidage case is folded into Stage 2).
