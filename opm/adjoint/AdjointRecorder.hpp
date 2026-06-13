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
 * \brief Records per-substep simulation state during a forward run for
 *        later adjoint/replay processing.
 *
 * Storage philosophy: write too much at first, optimize later. Every
 * accepted substep stores a full simulator snapshot (via the same
 * serializeOp machinery as --save-step) plus, optionally, the converged
 * residual and Jacobian for replay verification.
 */
#ifndef OPM_ADJOINT_RECORDER_HPP
#define OPM_ADJOINT_RECORDER_HPP

#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/models/discretization/common/fvbaseparameters.hh>
#include <opm/models/utils/parametersystem.hpp>
#include <opm/models/utils/propertysystem.hh>

#include <opm/adjoint/AdjointMeta.hpp>
#include <opm/adjoint/AdjointParameters.hpp>
#include <opm/adjoint/AdjointWellModel.hpp>
#include <opm/adjoint/AdjointStorage.hpp>
#include <opm/adjoint/AdjointSystemIO.hpp>
#include <opm/simulators/timestepping/SimulatorReport.hpp>

#include <memory>
#include <sstream>
#include <string>

namespace Opm {

template<class TypeTag>
class AdjointRecorder
{
public:
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;

    explicit AdjointRecorder(Simulator& simulator)
        : simulator_(simulator)
        , config_(AdjointConfig::fromParameters())
    {
        std::string path = config_.path;
        if (path.empty()) {
            const auto& ioConfig = simulator_.vanguard().eclState().getIOConfig();
            path = AdjointArchive::defaultPath(ioConfig.getOutputDir(),
                                               simulator_.vanguard().caseName());
        }
        archive_ = std::make_unique<AdjointArchive>(
            path, AdjointArchive::Mode::Write,
            simulator_.vanguard().gridView().comm());

        meta_.systemSaved = config_.saveSystem;
        meta_.storageCacheEnabled = Parameters::Get<Parameters::EnableStorageCache>();
        meta_.caseName = simulator_.vanguard().caseName();

        checkConfiguration_();
        OpmLog::info("Adjoint recording enabled; archive: " + path +
                     (archive_->usesHdf5() ? " (HDF5)" : " (directory store)"));
    }

    //! \brief Snapshot the initial state; call after the well model has
    //!        been set up for report step 0 (beginEpisode) but before the
    //!        first solve.
    void recordInitial()
    {
        // Sync the internal WGState copies so the snapshot can later be
        // read against a prepareDeserialize()'d well model in replay.
        normalizeWGStateForSerialization(simulator_.problem().wellModel());
        archive_->write(simulator_, AdjointGroups::initial(), "simulator");
    }

    //! \brief Record an accepted substep; call after problem.endTimeStep()
    //!        and before the substep timer advances.
    //! \param startTime Simulation time at the start of the substep [s].
    //! \param dt Substep length [s].
    //! \param substepInReport Substep counter within the current report step.
    //! \param report Solver report of this substep.
    void recordSubstep(double startTime,
                       double dt,
                       int substepInReport,
                       const SimulatorReportSingle& report)
    {
        const int k = static_cast<int>(meta_.substeps.size());
        const std::string group = AdjointGroups::substep(k);

        archive_->write(simulator_, group, "simulator");
        auto& wellModel = simulator_.problem().wellModel();
        archive_->write(wellModel, group, "wgstate");

        if (config_.saveSystem) {
            recordSystem_(group);
        }

        AdjointSubstepMeta meta;
        meta.globalIdx = k;
        meta.reportStep = simulator_.episodeIndex();
        meta.substepInReport = substepInReport;
        meta.startTime = startTime;
        meta.dt = dt;
        meta.newtonIterations = report.total_newton_iterations;
        meta_.substeps.push_back(meta);
    }

    //! \brief Write the archive metadata; call once at the end of the run.
    void finalize()
    {
        archive_->write(meta_, AdjointGroups::meta(), "meta");

        std::ostringstream params;
        Parameters::printValues(params);
        archive_->writeText(AdjointGroups::meta(), "parameters", params.str());

        OpmLog::info("Adjoint recording finished: " +
                     std::to_string(meta_.substeps.size()) + " substeps stored");
    }

    int numRecordedSubsteps() const
    { return static_cast<int>(meta_.substeps.size()); }

private:
    //! \brief Validity guard, warnings only: the warning list is the
    //!        worklist; nothing is refused at this stage.
    void checkConfiguration_()
    {
        if (meta_.storageCacheEnabled) {
            OpmLog::warning("Adjoint recording: the storage cache is enabled "
                            "(--enable-storage-cache=true). The replay driver "
                            "currently assumes the cache-free storage term; run "
                            "the forward simulation with "
                            "--enable-storage-cache=false for exact replay.");
        }
        if (simulator_.gridView().comm().size() > 1) {
            OpmLog::warning("Adjoint recording: running with more than one MPI "
                            "process is untested; replay is serial-first.");
        }
        if (Parameters::Get<Parameters::ThreadsPerProcess>() > 1) {
            OpmLog::warning("Adjoint recording: more than one thread per process "
                            "may prevent bitwise reproduction in replay; use "
                            "--threads-per-process=1 for the strictest tests.");
        }
        if (!config_.saveSystem) {
            OpmLog::info("Adjoint recording: residual/Jacobian dumps disabled "
                         "(enable with --adjoint-save-system=true for replay "
                         "verification).");
        }
    }

    void recordSystem_(const std::string& group)
    {
        const auto& linearizer = simulator_.model().linearizer();
        auto residual = AdjointVectorDump::from(linearizer.residual());
        archive_->write(residual, group + "/system", "residual");
        auto jacobian = AdjointMatrixDump::from(linearizer.jacobian().istlMatrix());
        archive_->write(jacobian, group + "/system", "jacobian");
    }

    Simulator& simulator_;
    AdjointConfig config_;
    std::unique_ptr<AdjointArchive> archive_;
    AdjointMeta meta_;
};

} // namespace Opm

#endif // OPM_ADJOINT_RECORDER_HPP
