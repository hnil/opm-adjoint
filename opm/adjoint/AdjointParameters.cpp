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

#include <config.h>
#include <opm/adjoint/AdjointParameters.hpp>

#include <opm/models/utils/parametersystem.hpp>

namespace Opm {

AdjointConfig AdjointConfig::fromParameters()
{
    AdjointConfig config;
    config.save = Parameters::Get<Parameters::AdjointSave>();
    config.mode = Parameters::Get<Parameters::AdjointMode>();
    config.saveSystem = Parameters::Get<Parameters::AdjointSaveSystem>();
    config.path = Parameters::Get<Parameters::AdjointFile>();
    config.replayTolerance = Parameters::Get<Parameters::AdjointReplayTolerance>();
    config.replayAbsTolerance = Parameters::Get<Parameters::AdjointReplayAbsTolerance>();
    return config;
}

void registerAdjointParameters()
{
    Parameters::Register<Parameters::AdjointSave>
        ("Record per-substep simulation state for adjoint/replay runs");
    Parameters::Register<Parameters::AdjointMode>
        ("Adjoint run mode: empty for a normal forward run, 'replay' to "
         "walk backwards over a recorded forward run and verify that its "
         "linearizations are reproduced");
    Parameters::Register<Parameters::AdjointSaveSystem>
        ("Also store the converged residual and Jacobian of every accepted "
         "substep in the adjoint archive (verification data for replay tests)");
    Parameters::Register<Parameters::AdjointFile>
        ("Path of the adjoint archive. Empty means <output-dir>/<CASENAME>.ADJOINT[.h5]. "
         "A path ending in .h5/.hdf5 selects the HDF5 backend, anything else "
         "a plain directory store");
    Parameters::Register<Parameters::AdjointReplayTolerance>
        ("Relative tolerance for comparing re-linearized systems against "
         "stored ones in adjoint replay");
    Parameters::Register<Parameters::AdjointReplayAbsTolerance>
        ("Absolute noise floor for the adjoint replay comparison; entry "
         "differences below this value count as rounding noise");
}

} // namespace Opm
