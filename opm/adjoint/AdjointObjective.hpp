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
 * \brief First objective functions for the adjoint solver.
 *
 * Per-substep sum objectives J = sum_k J_k(x_k) (the Jutul-style
 * interface). v1 ships a single deliberately simple reservoir-state
 * objective used to validate the full adjoint chain against finite
 * differences:
 *
 *   pressure-average: J = (1/N) sum_i x_i[pressureIdx] at the final
 *   substep — a pure function of the final solution vector, with no
 *   explicit dependence on the parameters or on well quantities, so the
 *   exact dJ/dx is trivial and the FD comparison is clean.
 *
 * Well-curve matching objectives (rates/BHP vs a reference summary via
 * ESmry) follow in the next step; they additionally require the
 * well-equation coupling in the right-hand side.
 */
#ifndef OPM_ADJOINT_OBJECTIVE_HPP
#define OPM_ADJOINT_OBJECTIVE_HPP

#include <opm/models/utils/propertysystem.hh>

#include <cstddef>

namespace Opm {

//! \brief J = average of the pressure primary variable over all cells at
//!        the final substep.
template<class TypeTag>
class PressureAverageObjective
{
public:
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Indices = GetPropType<TypeTag, Properties::Indices>;

    //! \brief Value contribution of substep \p k (solution(0) must hold x_k).
    Scalar stepValue(const Simulator& simulator, int k, int numSubsteps) const
    {
        if (k != numSubsteps - 1) {
            return 0.0;
        }
        const auto& solution = simulator.model().solution(/*timeIdx=*/0);
        Scalar sum = 0.0;
        for (std::size_t i = 0; i < solution.size(); ++i) {
            sum += solution[i][Indices::pressureSwitchIdx];
        }
        return sum / solution.size();
    }

    //! \brief Add dJ_k/dx_k to \p gradient (a GlobalEqVector-like
    //!        container of FieldVector blocks).
    template<class GradientVector>
    void addStepGradient(const Simulator& simulator,
                         int k,
                         int numSubsteps,
                         GradientVector& gradient) const
    {
        if (k != numSubsteps - 1) {
            return;
        }
        const std::size_t numCells = simulator.model().solution(0).size();
        for (std::size_t i = 0; i < numCells; ++i) {
            gradient[i][Indices::pressureSwitchIdx] += 1.0 / numCells;
        }
    }
};

} // namespace Opm

#endif // OPM_ADJOINT_OBJECTIVE_HPP
