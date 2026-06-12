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
 * \brief Well model used by the adjoint module.
 *
 * Thin extension of BlackoilWellModel that exposes the WGState
 * synchronization needed to make pre-step snapshots deserializable
 * (commitWGState is public upstream, updateNupcolWGState protected).
 * Keeping this in a derived class means the module builds against
 * unmodified opm-simulators master.
 */
#ifndef OPM_ADJOINT_WELL_MODEL_HPP
#define OPM_ADJOINT_WELL_MODEL_HPP

#include <opm/simulators/wells/BlackoilWellModel.hpp>

namespace Opm {

template<class TypeTag>
class AdjointWellModel : public BlackoilWellModel<TypeTag>
{
    using ParentType = BlackoilWellModel<TypeTag>;

public:
    using ParentType::ParentType;

    //! \brief Synchronize the internal WGState copies (last_valid, nupcol)
    //!        with the active one so that a snapshot serialized before the
    //!        first solve of a report step can later be deserialized
    //!        against a prepareDeserialize()'d well model.
    //!
    //! Both copies would be assigned the same values during the first
    //! iteration of the step anyway.
    void normalizeWGStateForSerialization()
    {
        this->commitWGState();
        this->updateNupcolWGState();
    }
};

} // namespace Opm

#endif // OPM_ADJOINT_WELL_MODEL_HPP
