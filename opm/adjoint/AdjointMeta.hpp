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
 * \brief Metadata describing the contents of an adjoint archive.
 */
#ifndef OPM_ADJOINT_META_HPP
#define OPM_ADJOINT_META_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace Opm {

//! \brief Per-substep record in the adjoint archive.
//!
//! One entry per *accepted* substep, in forward order; failed/chopped
//! attempts are never recorded. Substep k stores the simulator snapshot at
//! the END of the substep, i.e. solution(0) = x_k and solution(1) = x_{k-1}.
struct AdjointSubstepMeta
{
    int globalIdx{-1};        //!< global substep counter (0-based over whole run)
    int reportStep{-1};       //!< report step (episode index) this substep belongs to
    int substepInReport{-1};  //!< substep counter within the report step
    double startTime{0.0};    //!< simulation time at the start of the substep [s]
    double dt{0.0};           //!< substep length [s]
    unsigned newtonIterations{0}; //!< Newton iterations used in the forward solve

    template<class Serializer>
    void serializeOp(Serializer& serializer)
    {
        serializer(globalIdx);
        serializer(reportStep);
        serializer(substepInReport);
        serializer(startTime);
        serializer(dt);
        serializer(newtonIterations);
    }

    bool operator==(const AdjointSubstepMeta& rhs) const
    {
        return globalIdx == rhs.globalIdx &&
               reportStep == rhs.reportStep &&
               substepInReport == rhs.substepInReport &&
               startTime == rhs.startTime &&
               dt == rhs.dt &&
               newtonIterations == rhs.newtonIterations;
    }
};

//! \brief Archive-level metadata, written once at the end of the forward run.
struct AdjointMeta
{
    static constexpr int currentSchemaVersion = 1;

    int schemaVersion{currentSchemaVersion};
    bool systemSaved{false};        //!< residual/Jacobian dumps present
    bool storageCacheEnabled{true}; //!< EnableStorageCache of the forward run
    std::string caseName{};
    std::vector<AdjointSubstepMeta> substeps;

    template<class Serializer>
    void serializeOp(Serializer& serializer)
    {
        serializer(schemaVersion);
        serializer(systemSaved);
        serializer(storageCacheEnabled);
        serializer(caseName);
        serializer(substeps);
    }

    bool operator==(const AdjointMeta& rhs) const
    {
        return schemaVersion == rhs.schemaVersion &&
               systemSaved == rhs.systemSaved &&
               storageCacheEnabled == rhs.storageCacheEnabled &&
               caseName == rhs.caseName &&
               substeps == rhs.substeps;
    }
};

namespace AdjointGroups {

//! Group holding the snapshot of the initial state (before the first substep).
inline std::string initial()
{ return "/adjoint/initial"; }

//! Group holding all data of accepted substep \p k.
inline std::string substep(int k)
{ return "/adjoint/substep/" + std::to_string(k); }

//! Group holding archive metadata.
inline std::string meta()
{ return "/adjoint"; }

} // namespace AdjointGroups

} // namespace Opm

#endif // OPM_ADJOINT_META_HPP
