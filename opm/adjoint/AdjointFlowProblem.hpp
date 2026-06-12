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
 * \brief Flow problem variant that records per-substep adjoint state.
 *
 * All recording hooks live in lifecycle overrides of the problem class
 * (the geomech extension pattern), so the module builds against
 * unmodified opm-simulators master:
 *  - beginEpisode(): create the recorder lazily; snapshot the initial
 *    state after the wells of report step 0 are set up;
 *  - endTimeStep(): called exactly once per *accepted* substep (both in
 *    the adaptive substep loop and the non-adaptive path) — record the
 *    substep snapshot after the parent has finished its end-of-step
 *    processing;
 *  - setSubStepReport()/finalizeOutput() are non-virtual upstream but are
 *    always invoked on the concrete Problem type from templated code, so
 *    name hiding is sufficient.
 */
#ifndef OPM_ADJOINT_FLOW_PROBLEM_HPP
#define OPM_ADJOINT_FLOW_PROBLEM_HPP

#include <opm/models/utils/parametersystem.hpp>
#include <opm/models/utils/propertysystem.hh>

#include <opm/simulators/flow/FlowProblemBlackoil.hpp>
#include <opm/simulators/timestepping/SimulatorReport.hpp>

#include <opm/adjoint/AdjointParameters.hpp>
#include <opm/adjoint/AdjointRecorder.hpp>

#include <memory>

namespace Opm {

template<class TypeTag>
class AdjointFlowProblem : public FlowProblemBlackoil<TypeTag>
{
    using ParentType = FlowProblemBlackoil<TypeTag>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;

public:
    explicit AdjointFlowProblem(Simulator& simulator)
        : ParentType(simulator)
    {}

    static void registerParameters()
    {
        ParentType::registerParameters();
        registerAdjointParameters();
    }

    void beginEpisode() override
    {
        ParentType::beginEpisode();
        substepInEpisode_ = 0;

        if (!recorderInitialized_) {
            recorderInitialized_ = true;
            // Only record in forward runs; the replay/adjoint stages reuse
            // this problem class but must not overwrite the archive.
            if (Parameters::Get<Parameters::AdjointSave>() &&
                Parameters::Get<Parameters::AdjointMode>().empty()) {
                recorder_ = std::make_unique<AdjointRecorder<TypeTag>>(this->simulator());
            }
        }
        if (recorder_ && !initialRecorded_ && this->simulator().episodeIndex() == 0) {
            // The well/group state is now in its per-episode condition,
            // so the snapshot is deserializable via prepareDeserialize(0).
            recorder_->recordInitial();
            initialRecorded_ = true;
        }
    }

    void endTimeStep() override
    {
        ParentType::endTimeStep();
        if (recorder_) {
            const auto& simulator = this->simulator();
            recorder_->recordSubstep(simulator.time(),
                                     simulator.timeStepSize(),
                                     substepInEpisode_,
                                     lastSubStepReport_);
            ++substepInEpisode_;
        }
    }

    //! \brief Capture the substep report (Newton iterations for the
    //!        archive metadata); hides the non-virtual parent method,
    //!        which is always called on the concrete Problem type.
    void setSubStepReport(const SimulatorReportSingle& report)
    {
        lastSubStepReport_ = report;
        ParentType::setSubStepReport(report);
    }

    //! \brief Write the archive metadata at the end of the run (hides the
    //!        non-virtual parent method; called from the simulator's
    //!        finalize on the concrete Problem type).
    void finalizeOutput()
    {
        if (recorder_) {
            recorder_->finalize();
            recorder_.reset();
        }
        ParentType::finalizeOutput();
    }

private:
    std::unique_ptr<AdjointRecorder<TypeTag>> recorder_;
    SimulatorReportSingle lastSubStepReport_{};
    int substepInEpisode_{0};
    bool recorderInitialized_{false};
    bool initialRecorded_{false};
};

} // namespace Opm

#endif // OPM_ADJOINT_FLOW_PROBLEM_HPP
