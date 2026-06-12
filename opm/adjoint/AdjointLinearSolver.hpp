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
 * \brief Linear solver for the per-timestep adjoint systems J^T lambda = rhs.
 *
 * v1 strategy (see adjoint_plan.md): transpose the assembled matrix
 * explicitly and use a direct solver (UMFPACK). This removes one error
 * source while the adjoint recursion itself is validated; an iterative
 * path (FlexibleSolver with the existing transposed-CPR preconditioner
 * "cprt" on the explicitly transposed matrix) can be added behind the
 * same interface later. Serial only.
 */
#ifndef OPM_ADJOINT_LINEAR_SOLVER_HPP
#define OPM_ADJOINT_LINEAR_SOLVER_HPP

#include <opm/common/ErrorMacros.hpp>

#include <opm/adjoint/transposeMatrix.hpp>

#if HAVE_SUITESPARSE_UMFPACK
#include <dune/istl/umfpack.hh>
#endif

#include <dune/istl/solver.hh>

#include <stdexcept>

namespace Opm {

//! \brief Direct solver for transposed systems.
//!
//! \tparam Matrix BCRS matrix with square dense blocks (double scalar).
//! \tparam Vector Block vector matching the matrix block size.
template<class Matrix, class Vector>
class AdjointLinearSolver
{
public:
    //! \brief Solve A^T x = b.
    //!
    //! The transpose is formed explicitly; the right-hand side is taken
    //! by value because direct solvers overwrite it.
    void solveTransposed(const Matrix& matrix, Vector rhs, Vector& solution)
    {
#if HAVE_SUITESPARSE_UMFPACK
        const auto transposed = transposeBlockMatrix(matrix);

        solution.resize(rhs.size());
        solution = 0.0;

        Dune::UMFPack<Matrix> solver(transposed, /*verbose=*/0);
        Dune::InverseOperatorResult result;
        solver.apply(solution, rhs, result);

        if (!result.converged) {
            OPM_THROW(std::runtime_error,
                      "UMFPACK failed to solve the transposed (adjoint) system");
        }
#else
        static_cast<void>(matrix);
        static_cast<void>(rhs);
        static_cast<void>(solution);
        OPM_THROW(std::runtime_error,
                  "AdjointLinearSolver requires UMFPACK (SuiteSparse); "
                  "this build has no SuiteSparse support.");
#endif
    }
};

} // namespace Opm

#endif // OPM_ADJOINT_LINEAR_SOLVER_HPP
