/*
  Copyright 2026 SINTEF Digital, Mathematics and Cybernetics.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
/*!
 * \file
 * \brief Backward replay of a recorded forward simulation.
 *
 * For every accepted substep k (walked backwards), the converged
 * linearization of the forward run is recreated from the stored snapshots
 * and compared against the stored residual/Jacobian (when present). This
 * is test level T2/T3 of the adjoint plan and the foundation the adjoint
 * solve builds on: after replayStep(k) returns, the linearizer holds the
 * converged system of forward substep k.
 *
 * v1 restrictions: serial runs, --enable-storage-cache=false (the replay
 * relies on the cache-free storage term, where the residual is a pure
 * function R(x_k, x_{k-1})).
 */
#ifndef OPM_ADJOINT_REPLAY_HPP
#define OPM_ADJOINT_REPLAY_HPP

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/models/discretization/common/fvbaseparameters.hh>
#include <opm/models/utils/parametersystem.hpp>
#include <opm/models/utils/propertysystem.hh>

#include <opm/adjoint/AdjointMeta.hpp>
#include <opm/adjoint/AdjointParameters.hpp>
#include <opm/adjoint/AdjointStorage.hpp>
#include <opm/adjoint/AdjointSystemIO.hpp>

#include <dune/common/fmatrix.hh>

#include <vector>

#include <fmt/format.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace Opm {

template<class TypeTag>
class AdjointReplay
{
public:
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using SolutionVector = GetPropType<TypeTag, Properties::SolutionVector>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using LocalResidual = GetPropType<TypeTag, Properties::LocalResidual>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;

    static constexpr int numEq = getPropValue<TypeTag, Properties::NumEq>();
    //! Dense block of dR_k/dx_{k-1} for one cell (block-diagonal cross term).
    using BdiagBlock = Dune::FieldMatrix<Scalar, numEq, numEq>;

    explicit AdjointReplay(Simulator& simulator)
        : simulator_(simulator)
        , config_(AdjointConfig::fromParameters())
    {
        if (Parameters::Get<Parameters::EnableStorageCache>()) {
            OPM_THROW(std::runtime_error,
                      "Adjoint replay requires the cache-free storage term; "
                      "run with --enable-storage-cache=false (and record the "
                      "forward run with the same option).");
        }

        std::string path = config_.path;
        if (path.empty()) {
            const auto& ioConfig = simulator_.vanguard().eclState().getIOConfig();
            path = AdjointArchive::defaultPath(ioConfig.getOutputDir(),
                                               simulator_.vanguard().caseName());
        }
        archive_ = std::make_unique<AdjointArchive>(
            path, AdjointArchive::Mode::Read,
            simulator_.vanguard().gridView().comm());
        archive_->read(meta_, AdjointGroups::meta(), "meta");

        if (meta_.schemaVersion != AdjointMeta::currentSchemaVersion) {
            OPM_THROW(std::runtime_error,
                      "Adjoint archive schema version " +
                      std::to_string(meta_.schemaVersion) + " does not match " +
                      std::to_string(AdjointMeta::currentSchemaVersion));
        }
        if (meta_.storageCacheEnabled) {
            OpmLog::warning("The forward run was recorded with the storage "
                            "cache enabled; exact replay is not expected. "
                            "Re-run the forward simulation with "
                            "--enable-storage-cache=false.");
        }

        OpmLog::info(fmt::format("Adjoint replay: {} substeps in archive {}",
                                 meta_.substeps.size(), path));
    }

    //! \brief Walk backwards over all substeps, re-linearize and compare.
    //! \return EXIT_SUCCESS when every substep matched within tolerance.
    int run()
    {
        const int numSubsteps = static_cast<int>(meta_.substeps.size());
        int failed = 0;
        double worstResidual = 0.0;
        double worstJacobian = 0.0;

        for (int k = numSubsteps - 1; k >= 0; --k) {
            if (!replayStep(k, worstResidual, worstJacobian)) {
                ++failed;
            }
        }

        if (meta_.systemSaved) {
            OpmLog::info(fmt::format(
                "Adjoint replay finished: {}/{} substeps within tolerance {:.1e} "
                "(worst residual rel diff {:.3e}, worst Jacobian rel diff {:.3e})",
                numSubsteps - failed, numSubsteps, config_.replayTolerance,
                worstResidual, worstJacobian));
        } else {
            OpmLog::info("Adjoint replay finished (no stored systems to compare "
                         "against; record with --adjoint-save-system=true for "
                         "verification)");
        }
        return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    //! \brief Recreate the converged linearization of forward substep \p k.
    //!
    //! On successful return the linearizer holds the converged residual and
    //! Jacobian of substep k (the state the adjoint solve needs).
    //! \return false if a stored system exists and the comparison exceeded
    //!         the tolerance.
    bool replayStep(int k, double& worstResidual, double& worstJacobian)
    {
        const auto& meta = meta_.substeps[k];
        const auto& schedule = simulator_.vanguard().schedule();

        // Phase A: extract the converged solution x_k from snapshot k.
        loadSnapshot_(k, meta.reportStep);
        SolutionVector convergedSolution = model_().solution(/*timeIdx=*/0);

        // Phase B: restore the state at the end of substep k-1 (the initial
        // state for k == 0).
        const int prevReport = (k > 0) ? meta_.substeps[k - 1].reportStep
                                       : meta.reportStep;
        loadSnapshot_(k - 1, prevReport);

        // Report-step context, mirroring SimulatorFullyImplicit::runStep and
        // the --load-step path (episode setup after the restore, then
        // beginEpisode to rebuild the well container from the restored
        // well/group state).
        //
        // NOTE: for substeps in the middle of a report step this re-applies
        // the start-of-episode schedule processing, which the forward run
        // performed only once per report step (with the state of the last
        // substep of the previous report step). For simple decks
        // (constant controls) this is idempotent; control switching during
        // an episode is a documented v1 risk - the T2/T3 comparisons below
        // detect any resulting drift.
        simulator_.startNextEpisode(simulator_.startTime() +
                                        schedule.seconds(meta.reportStep),
                                    schedule.stepLength(meta.reportStep));
        simulator_.setEpisodeIndex(meta.reportStep);
        problem_().beginEpisode();

        // prepareStep mirror (NonlinearSystem::prepareStep).
        model_().advanceTimeLevel();
        simulator_.setTime(meta.startTime);
        simulator_.setTimeStepSize(meta.dt);
        problem_().resetIterationForNewTimestep();
        problem_().beginTimeStep();

        // Iteration-0 assembly: triggers the well model's per-step setup
        // (prepareTimeStep -> explicit quantities from state k-1) exactly
        // like the forward run's initial linearization.
        problem_().beginIteration();
        model_().linearizer().linearizeDomain();
        problem_().endIteration();
        problem_().markTimestepInitialized();
        problem_().advanceIteration();

        // Milestone B cross term: at this point solution(0) = x_{k-1} and
        // the intensive quantities at time index 0 carry derivatives with
        // respect to x_{k-1}, so the block-diagonal
        // Bdiag_k = dR_k/dx_{k-1} = -(V/dt) dStorage/dx|x_{k-1}
        // is a cell-local evaluation (fluxes and sources are purely
        // implicit in the TPFA residual).
        if (computeBdiagEnabled_) {
            captureBdiag_(meta.dt);
        }

        // Install the converged state of substep k.
        model_().solution(/*timeIdx=*/0) = convergedSolution;
        wellModel_().prepareDeserialize(meta.reportStep);
        archive_->read(wellModel_(), AdjointGroups::substep(k), "wgstate");
        model_().invalidateAndUpdateIntensiveQuantities(/*timeIdx=*/0);

        // Final linearization at the converged point. Assemble the well
        // equations from the restored converged controls WITHOUT
        // re-deriving the control/group/network/guide-rate state (which
        // does not reproduce the forward's converged values and crashes on
        // shut wells); recompute only the well-local quantities.
        wellModel_().assembleWellEqGivenControls(meta.dt);
        model_().linearizer().linearizeDomain();

        return compareWithStored_(k, worstResidual, worstJacobian);
    }

    const AdjointMeta& meta() const
    { return meta_; }

    //! \brief Enable capturing of the cross-term blocks during replay.
    void setComputeBdiag(bool enable)
    { computeBdiagEnabled_ = enable; }

    //! \brief Cross-term blocks dR_k/dx_{k-1} of the last replayed substep.
    const std::vector<BdiagBlock>& bdiag() const
    { return bdiag_; }

    //! \brief Full restore of snapshot \p k (k == -1 means the initial
    //!        state). Public for objective-only evaluation.
    void loadSnapshot(int k, int reportStepForWells)
    { loadSnapshot_(k, reportStepForWells); }

    Simulator& simulator()
    { return simulator_; }

private:
    auto& model_()
    { return simulator_.model(); }

    auto& problem_()
    { return simulator_.problem(); }

    auto& wellModel_()
    { return simulator_.problem().wellModel(); }

    //! \brief Full restore of snapshot \p k (k == -1 means the initial state).
    void loadSnapshot_(int k, int reportStepForWells)
    {
        wellModel_().prepareDeserialize(reportStepForWells);
        const std::string group = (k < 0) ? AdjointGroups::initial()
                                          : AdjointGroups::substep(k);
        archive_->read(simulator_, group, "simulator");
        model_().invalidateAndUpdateIntensiveQuantities(/*timeIdx=*/0);
    }

    void captureBdiag_(double dt)
    {
        const std::size_t numCells = model_().numGridDof();
        bdiag_.resize(numCells);
        for (std::size_t globI = 0; globI < numCells; ++globI) {
            const auto& intQuants = model_().intensiveQuantities(globI, /*timeIdx=*/0);
            Dune::FieldVector<Evaluation, numEq> storage;
            LocalResidual::template computeStorage<Evaluation>(storage, intQuants);
            const Scalar factor = -model_().dofTotalVolume(globI) / dt;
            for (int eq = 0; eq < numEq; ++eq) {
                for (int pv = 0; pv < numEq; ++pv) {
                    bdiag_[globI][eq][pv] = factor * storage[eq].derivative(pv);
                }
            }
        }
    }

    bool compareWithStored_(int k, double& worstResidual, double& worstJacobian)
    {
        const auto& meta = meta_.substeps[k];
        auto liveResidual = AdjointVectorDump::from(model_().linearizer().residual());

        if (!meta_.systemSaved) {
            double norm = 0.0;
            for (const double value : liveResidual.values) {
                norm = std::max(norm, std::abs(value));
            }
            OpmLog::info(fmt::format(
                "replay substep {:4d} (report step {:3d}, dt {:.4g} days): "
                "max |residual| = {:.3e}", k, meta.reportStep,
                meta.dt / 86400.0, norm));
            return true;
        }

        const std::string group = AdjointGroups::substep(k) + "/system";

        AdjointVectorDump storedResidual;
        archive_->read(storedResidual, group, "residual");
        const auto residualDiff = compare(storedResidual, liveResidual,
                                          1e-300, config_.replayAbsTolerance);

        AdjointMatrixDump storedJacobian;
        archive_->read(storedJacobian, group, "jacobian");
        auto liveJacobian =
            AdjointMatrixDump::from(model_().linearizer().jacobian().istlMatrix());
        const auto jacobianDiff = compare(storedJacobian, liveJacobian,
                                          1e-300, config_.replayAbsTolerance);

        worstResidual = std::max(worstResidual, residualDiff.maxRelDiff);
        worstJacobian = std::max(worstJacobian, jacobianDiff.maxRelDiff);

        const bool pass = residualDiff.withinTolerance(config_.replayTolerance) &&
                          jacobianDiff.withinTolerance(config_.replayTolerance);

        OpmLog::info(fmt::format(
            "replay substep {:4d} (report step {:3d}, dt {:.4g} days): "
            "residual rel diff {:.3e} (abs {:.3e}), "
            "Jacobian rel diff {:.3e} (abs {:.3e}){}{}",
            k, meta.reportStep, meta.dt / 86400.0,
            residualDiff.maxRelDiff, residualDiff.maxAbsDiff,
            jacobianDiff.maxRelDiff, jacobianDiff.maxAbsDiff,
            pass ? "" : "  <-- EXCEEDS TOLERANCE",
            residualDiff.structureMatches && jacobianDiff.structureMatches
                ? "" : "  (structure mismatch)"));

        return pass;
    }

    Simulator& simulator_;
    AdjointConfig config_;
    std::unique_ptr<AdjointArchive> archive_;
    AdjointMeta meta_;
    bool computeBdiagEnabled_{false};
    std::vector<BdiagBlock> bdiag_;
};

} // namespace Opm

#endif // OPM_ADJOINT_REPLAY_HPP
