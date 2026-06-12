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
 * \brief Runtime parameters controlling adjoint state recording and replay.
 *
 * These parameters are registered by all flow variants (via
 * SimulatorFullyImplicit::registerParameters) but everything is off by
 * default; a normal run is unaffected.
 */
#ifndef OPM_ADJOINT_PARAMETERS_HPP
#define OPM_ADJOINT_PARAMETERS_HPP

#include <string>

namespace Opm::Parameters {

//! \brief Record per-substep simulation state for later adjoint/replay runs.
struct AdjointSave { static constexpr bool value = false; };

//! \brief Additionally store the converged residual and Jacobian of every
//!        accepted substep ("write too much" verification data for the
//!        replay tests).
struct AdjointSaveSystem { static constexpr bool value = false; };

//! \brief Run mode of the simulator with respect to adjoint processing:
//!        "" (normal forward run) or "replay" (backward replay of a
//!        recorded forward run instead of the time loop).
struct AdjointMode { static constexpr auto value = ""; };

//! \brief Path of the adjoint archive. Empty means
//!        <output-dir>/<CASENAME>.ADJOINT[.h5]. A path ending in .h5/.hdf5
//!        selects the HDF5 backend, anything else a plain directory store.
struct AdjointFile { static constexpr auto value = ""; };

//! \brief Linear solver for the transposed adjoint systems:
//!        umfpack (direct, default) | ilu0 | cpr | cprt | <file>.json
//!        (FlexibleSolver configuration).
struct AdjointLinearSolver { static constexpr auto value = "umfpack"; };

//! \brief Relative residual reduction for iterative adjoint solves.
//!        Tighter than forward solves: lambda accuracy multiplies dR/dm
//!        directly in the gradients.
struct AdjointLinearSolverReduction { static constexpr double value = 1e-14; };

//! \brief Iteration cap for iterative adjoint solves.
struct AdjointLinearSolverMaxIter { static constexpr int value = 300; };

//! \brief Verbosity for the adjoint linear solver (passed to
//!        FlexibleSolver; 0 = quiet).
struct AdjointLinearSolverVerbosity { static constexpr int value = 0; };

//! \brief Comma-separated list of end-point scaling parameters to
//!        compute gradients for (e.g. "SWL,KRW"); empty disables.
struct AdjointEndpoints { static constexpr auto value = ""; };

//! \brief Objective function for adjoint/gradient runs:
//!        "pressure-average" or "bhp:<WELL>".
struct AdjointObjective { static constexpr auto value = "pressure-average"; };

//! \brief Relative tolerance used when the replay driver compares the
//!        re-linearized systems against the stored ones.
struct AdjointReplayTolerance { static constexpr double value = 1e-12; };

//! \brief Absolute noise floor for the replay comparison: entries whose
//!        absolute difference is below this are counted as rounding noise
//!        and excluded from the relative criterion.
struct AdjointReplayAbsTolerance { static constexpr double value = 1e-9; };

} // namespace Opm::Parameters

namespace Opm {

//! \brief Settings for adjoint state recording, resolved from the
//!        parameter system.
struct AdjointConfig
{
    bool save{false};         //!< record per-substep state
    std::string mode{};       //!< "" (forward) or "replay"
    bool saveSystem{false};   //!< also dump residual/Jacobian per substep
    std::string path{};       //!< archive path ("" = derive from case name)
    double replayTolerance{1e-12};
    double replayAbsTolerance{1e-9};
    std::string objective{};  //!< objective specification
    std::string endpoints{};  //!< end-point parameter list
    std::string linearSolver{};  //!< adjoint linear solver spec
    double linearSolverReduction{1e-14};
    int linearSolverMaxIter{300};
    int linearSolverVerbosity{0};

    static AdjointConfig fromParameters();
};

//! \brief Register the adjoint runtime parameters.
void registerAdjointParameters();

} // namespace Opm

#endif // OPM_ADJOINT_PARAMETERS_HPP
