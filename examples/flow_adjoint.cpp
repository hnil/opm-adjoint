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
 * \brief flow_adjoint: blackoil simulator with adjoint state recording and
 *        backward replay, built as a downstream module against unmodified
 *        opm-simulators.
 *
 * Workflow (single binary):
 *   1. forward + record:
 *        flow_adjoint CASE.DATA --adjoint-save=true --adjoint-save-system=true \
 *            --enable-storage-cache=false --threads-per-process=1
 *   2. backward replay + verification:
 *        flow_adjoint CASE.DATA --adjoint-mode=replay \
 *            --enable-storage-cache=false --threads-per-process=1
 *
 * The TypeTag mirrors the production blackoil TPFA setup
 * (flow/flow_blackoil.cpp) and adds the adjoint problem/well-model
 * extensions via property overrides.
 */
#include "config.h"

#include <opm/material/common/ResetLocale.hpp>
#include <opm/grid/CpGrid.hpp>
#include <opm/simulators/flow/FlowMain.hpp>
#include <opm/simulators/flow/Main.hpp>

#include <opm/models/blackoil/blackoilconvectivemixingmodule.hh>
#include <opm/models/blackoil/blackoillocalresidualtpfa.hh>
#include <opm/models/discretization/common/tpfalinearizer.hh>

#include <opm/adjoint/AdjointFlowProblem.hpp>
#include <opm/adjoint/AdjointReplay.hpp>
#include <opm/adjoint/AdjointWellModel.hpp>

#include <cstring>
#include <memory>
#include <string>

namespace Opm::Properties {

namespace TTag {
struct FlowAdjointProblem {
    using InheritsFrom = std::tuple<FlowProblemTPFA>;
};
}

// Same discretization choices as the production blackoil simulator
// (flow/flow_blackoil.cpp).
template<class TypeTag>
struct Linearizer<TypeTag, TTag::FlowAdjointProblem>
{ using type = TpfaLinearizer<TypeTag>; };

template<class TypeTag>
struct LocalResidual<TypeTag, TTag::FlowAdjointProblem>
{ using type = BlackOilLocalResidualTPFA<TypeTag>; };

template<class TypeTag>
struct EnableDiffusion<TypeTag, TTag::FlowAdjointProblem>
{ static constexpr bool value = false; };

template<class TypeTag>
struct AvoidElementContext<TypeTag, TTag::FlowAdjointProblem>
{ static constexpr bool value = true; };

// Adjoint extensions, hooked in via the property system (no
// opm-simulators changes).
template<class TypeTag>
struct Problem<TypeTag, TTag::FlowAdjointProblem>
{ using type = AdjointFlowProblem<TypeTag>; };

template<class TypeTag>
struct WellModel<TypeTag, TTag::FlowAdjointProblem>
{ using type = AdjointWellModel<TypeTag>; };

} // namespace Opm::Properties

namespace Opm {

//! \brief Main variant dispatching between the normal forward run (with
//!        recording handled by AdjointFlowProblem) and the backward
//!        replay driver.
class AdjointMain : public Main
{
public:
    using Main::Main;

    template<class TypeTag>
    int runForward()
    {
        int exitCode = EXIT_SUCCESS;
        if (initialize_<TypeTag>(exitCode)) {
            // Mirrors Main::dispatchStatic_.
            this->setupVanguard();
            FlowMain<TypeTag> flowMain(argc_, argv_, outputCout_, outputFiles_);
            exitCode = flowMain.execute();
        }
        return exitCode;
    }

    template<class TypeTag>
    int runReplay()
    {
        int exitCode = EXIT_SUCCESS;
        if (initialize_<TypeTag>(exitCode)) {
            this->setupVanguard();
            FlowMain<TypeTag> flowMain(argc_, argv_, outputCout_, outputFiles_);
            // Full production setup without running the time loop.
            const int status = flowMain.executeInitStep();
            if (status == EXIT_SUCCESS) {
                AdjointReplay<TypeTag> replay(*flowMain.getSimulatorPtr());
                exitCode = replay.run();
            } else {
                exitCode = status;
            }
        }
        return exitCode;
    }
};

} // namespace Opm

int main(int argc, char** argv)
{
    using TypeTag = Opm::Properties::TTag::FlowAdjointProblem;

    // The run mode must be known before the parameter system is set up,
    // so peek at the raw command line.
    bool replay = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--adjoint-mode=replay", 21) == 0) {
            replay = true;
        }
    }

    auto mainObject = std::make_unique<Opm::AdjointMain>(argc, argv);
    const int exitCode = replay ? mainObject->runReplay<TypeTag>()
                                : mainObject->runForward<TypeTag>();
    // Destruct mainObject as the destructor calls MPI_Finalize!
    mainObject.reset();
    return exitCode;
}
