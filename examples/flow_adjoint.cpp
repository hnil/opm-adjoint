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
#include <opm/simulators/flow/TTagFlowProblemTPFA.hpp>

#include <opm/models/blackoil/blackoilconvectivemixingmodule.hh>
#include <opm/models/blackoil/blackoillocalresidualtpfa.hh>
#include <opm/models/discretization/common/tpfalinearizer.hh>

#include <opm/adjoint/AdjointFlowProblem.hpp>
#include <opm/adjoint/AdjointReplay.hpp>
#include <opm/adjoint/AdjointSolver.hpp>

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

// Adjoint extension, hooked in via the property system (no
// opm-simulators changes).
template<class TypeTag>
struct Problem<TypeTag, TTag::FlowAdjointProblem>
{ using type = AdjointFlowProblem<TypeTag>; };

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
    int runReplay(const std::string& mode)
    {
        int exitCode = EXIT_SUCCESS;
        if (initialize_<TypeTag>(exitCode)) {
            this->setupVanguard();
            FlowMain<TypeTag> flowMain(argc_, argv_, outputCout_, outputFiles_);
            // Full production setup without running the time loop.
            const int status = flowMain.executeInitStep();
            if (status != EXIT_SUCCESS) {
                return status;
            }
            if (mode == "replay") {
                AdjointReplay<TypeTag> replay(*flowMain.getSimulatorPtr());
                exitCode = replay.run();
            } else if (mode == "gradient") {
                AdjointSolver<TypeTag> solver(*flowMain.getSimulatorPtr());
                exitCode = solver.run();
            } else { // "objective"
                AdjointSolver<TypeTag> solver(*flowMain.getSimulatorPtr());
                exitCode = solver.evaluateObjectiveOnly();
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
    std::string mode;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--adjoint-mode=", 15) == 0) {
            mode = argv[i] + 15;
        }
    }

    // Adjoint-appropriate defaults, injected as command-line arguments so
    // they sit below any value the user passes explicitly (an injected
    // arg only takes effect when the user did not set that option):
    //   - the Schur step (addWellContributions) needs the well coupling
    //     in the matrix sparsity, so default --matrix-add-well-contributions
    //     on (single-perforation decks are unaffected; multi-perforation
    //     decks would otherwise crash without it);
    //   - with that flag on, flow auto-promotes the forward solver "cpr"
    //     to "cprw" (CPR with the well system in the matrix), which is not
    //     registered in this build; "ilu0" is robust on any problem size
    //     (incl. the 3-cell test deck) and is not promoted, so default the
    //     forward solver to it. The adjoint backward solves use their own
    //     --adjoint-linear-solver and are unaffected.
    // Done via argv injection (not Parameters::SetDefault) because the
    // model/linalg parameters are registered after the problem, so a
    // SetDefault in the problem's registerParameters runs too early.
    std::vector<std::string> injected;
    auto hasOption = [argc, argv](const char* opt) {
        const std::size_t n = std::strlen(opt);
        for (int i = 1; i < argc; ++i) {
            if (std::strncmp(argv[i], opt, n) == 0) {
                return true;
            }
        }
        return false;
    };
    if (!hasOption("--matrix-add-well-contributions")) {
        injected.emplace_back("--matrix-add-well-contributions=true");
    }
    if (!hasOption("--linear-solver")) {
        injected.emplace_back("--linear-solver=ilu0");
    }

    std::vector<char*> args(argv, argv + argc);
    for (auto& opt : injected) {
        args.push_back(opt.data());
    }
    int newArgc = static_cast<int>(args.size());
    char** newArgv = args.data();

    auto mainObject = std::make_unique<Opm::AdjointMain>(newArgc, newArgv);
    const int exitCode = mode.empty() ? mainObject->runForward<TypeTag>()
                                      : mainObject->runReplay<TypeTag>(mode);
    // Destruct mainObject as the destructor calls MPI_Finalize!
    mainObject.reset();
    return exitCode;
}
