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
 * The transpose is always formed explicitly (one structural pass plus a
 * value copy - negligible next to the solve), which makes every existing
 * solver and preconditioner usable unchanged. Two paths behind one
 * interface (--adjoint-linear-solver):
 *
 *  - "umfpack" (default): direct solve, removes one error source while
 *    the adjoint recursion itself is validated;
 *  - iterative (the plan's v1.5): FlexibleSolver on the transposed
 *    matrix, configured by preset name "ilu0" / "cpr" / "cprt" (flow's
 *    own property-tree defaults via setupILU/setupCPR) or a FlexibleSolver
 *    JSON file. "cprt" is CPR built for transposed systems
 *    (PressureTransferPolicy<transpose=true> + quasi-IMPES weights with
 *    transpose=true), i.e. the canonical adjoint preconditioner; "cpr"
 *    on the transposed matrix is kept for comparison. trueimpes weights
 *    need simulator internals and are not supported here - quasiimpes
 *    only.
 *
 * Parallel runs use the iterative path with the ghost-last operator
 * (interior rows only + copyOwnerToAll): the locally transposed matrix
 * has correct interior rows (see AdjointParallel.hpp for why), the
 * ghost rows are invalidated like in the forward ISTLSolver (except for
 * paroverilu0, which handles the overlap itself). UMFPACK remains
 * single-rank.
 */
#ifndef OPM_ADJOINT_LINEAR_SOLVER_HPP
#define OPM_ADJOINT_LINEAR_SOLVER_HPP

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/WellOperators.hpp>
#include <opm/simulators/linalg/FlowLinearSolverParameters.hpp>
#include <opm/simulators/linalg/getQuasiImpesWeights.hpp>
#include <opm/simulators/linalg/setupPropertyTree.hpp>

#include <opm/adjoint/transposeMatrix.hpp>

#if HAVE_SUITESPARSE_UMFPACK
#include <dune/istl/umfpack.hh>
#endif

#include <dune/istl/operators.hh>
#include <dune/istl/solver.hh>
#if HAVE_MPI
#include <dune/istl/owneroverlapcopy.hh>
#endif

#include <fmt/format.h>

#include <functional>
#include <stdexcept>
#include <string>

namespace Opm {

//! \brief Solver for transposed systems A^T x = b.
//!
//! \tparam Matrix BCRS matrix with square dense blocks (double scalar).
//! \tparam Vector Block vector matching the matrix block size.
template<class Matrix, class Vector>
class AdjointLinearSolver
{
public:
    //! \brief Select solver and tolerances; must be called before solves.
    //!
    //! \param spec "umfpack", "ilu0", "cpr", "cprt", or a path to a
    //!             FlexibleSolver JSON configuration (*.json).
    //! \param pressureIndex pressure variable index for CPR-type
    //!                      preconditioners (blackoil: 1).
    void configure(const std::string& spec,
                   double reduction,
                   int maxIter,
                   int verbosity,
                   std::size_t pressureIndex)
    {
        spec_ = spec.empty() ? "umfpack" : spec;
        pressureIndex_ = pressureIndex;
        if (spec_ == "umfpack") {
            return;
        }

        FlowLinearSolverParameters params; // reset() defaults
        params.linear_solver_reduction_ = reduction;
        params.linear_solver_maxiter_ = maxIter;
        params.linear_solver_verbosity_ = verbosity;

        if (spec_.size() > 5 && spec_.substr(spec_.size() - 5) == ".json") {
            prm_ = PropertyTree(spec_);
        } else if (spec_ == "ilu0") {
            prm_ = setupILU(spec_, params);
        } else if (spec_ == "cpr" || spec_ == "cprt") {
            prm_ = setupCPR("cpr_quasiimpes", params);
            prm_.put("preconditioner.type", spec_);
        } else {
            OPM_THROW(std::runtime_error,
                      "Unknown --adjoint-linear-solver '" + spec_ +
                      "' (use umfpack, ilu0, cpr, cprt or a "
                      "FlexibleSolver *.json file)");
        }

        const auto precType =
            prm_.get<std::string>("preconditioner.type", "");
        usesCprWeights_ = (precType == "cpr" || precType == "cprt" ||
                           precType == "cprw" || precType == "cprwt");
        transposedWeights_ = (precType == "cprt" || precType == "cprwt");
        if (usesCprWeights_) {
            const auto weightsType =
                prm_.get<std::string>("preconditioner.weight_type",
                                      "quasiimpes");
            if (weightsType != "quasiimpes") {
                OPM_THROW(std::runtime_error,
                          "Adjoint CPR supports quasiimpes weights only "
                          "(trueimpes needs forward-simulator internals)");
            }
        }
    }

    //! \brief Solve A^T x = b.
    //!
    //! The right-hand side is taken by value because solvers overwrite it.
    void solveTransposed(const Matrix& matrix, Vector rhs, Vector& solution)
    {
        solution.resize(rhs.size());
        solution = 0.0;

        if (spec_ == "umfpack") {
            solveDirect_(matrix, rhs, solution);
        } else {
            solveIterative_(matrix, rhs, solution);
        }
        ++numSolves_;
    }

#if HAVE_MPI
    //! \brief Enable the parallel solve path (ghost-last operator).
    //!
    //! \param comm ISTL communication (owner/overlap/copy)
    //! \param overlapRows local indices of the ghost rows
    void setComm(const Dune::OwnerOverlapCopyCommunication<int, int>* comm,
                 const std::vector<int>* overlapRows)
    {
        comm_ = comm;
        overlapRows_ = overlapRows;
        if (comm_ && spec_ == "umfpack") {
            OPM_THROW(std::runtime_error,
                      "--adjoint-linear-solver=umfpack is a single-rank "
                      "direct solver; parallel adjoint runs need an "
                      "iterative solver (cpr, ilu0 or a *.json config)");
        }
        // cprt (transpose-CPR) falsely converges in parallel: it reports
        // convergence quickly but the lambda is wrong (verified against
        // serial and against cpr/ilu0 on SPE1). The transposed pressure
        // transfer has not been exercised in a parallel setting. cprt is
        // correct and fastest in serial; in parallel use cpr or ilu0.
        if (comm_ && (spec_ == "cprt" || spec_ == "cprwt" ||
                      transposedWeights_)) {
            OPM_THROW(std::runtime_error,
                      "--adjoint-linear-solver=cprt is not reliable in "
                      "parallel (false convergence of the transposed CPR). "
                      "Use cpr or ilu0 for parallel adjoint runs; cprt is "
                      "fine in serial.");
        }
    }
#endif

    //! \brief Iterations of the iterative path summed over all solves
    //!        (0 for umfpack).
    int totalIterations() const
    { return totalIterations_; }

    int numSolves() const
    { return numSolves_; }

    const std::string& spec() const
    { return spec_; }

private:
    void solveDirect_(const Matrix& matrix, Vector& rhs, Vector& solution)
    {
#if HAVE_SUITESPARSE_UMFPACK
        const auto transposed = transposeBlockMatrix(matrix);

        Dune::UMFPack<TransposedMatrix<Matrix>> solver(transposed, /*verbose=*/0);
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
                  "--adjoint-linear-solver=umfpack requires SuiteSparse; "
                  "this build has no SuiteSparse support. Use an iterative "
                  "solver (ilu0, cpr, cprt or a *.json config).");
#endif
    }

    void solveIterative_(const Matrix& matrix, Vector& rhs, Vector& solution)
    {
        Matrix transposed = transposeBlockMatrixSameType(matrix);

        std::function<Vector()> weightsCalculator;
        if (usesCprWeights_) {
            // mirror of ISTLSolver::getWeightsCalculator (quasiimpes
            // branch): weights from the matrix handed to the
            // preconditioner - here the transposed matrix - with the
            // transpose flag matching the cprt/cpr choice.
            const auto& matrixRef = transposed;
            const bool transposeFlag = transposedWeights_;
            const std::size_t pIdx = pressureIndex_;
            weightsCalculator = [&matrixRef, transposeFlag, pIdx]() {
                return Amg::getQuasiImpesWeights<Matrix, Vector>(
                    matrixRef, pIdx, transposeFlag,
                    /*enable_thread_parallel=*/false);
            };
        }

        Dune::InverseOperatorResult result;
#if HAVE_MPI
        if (comm_) {
            // ghost rows of the local transpose are incomplete; replace
            // them by identity like the forward solver does (paroverilu0
            // handles the overlap itself and keeps the assembled rows)
            const auto precType =
                prm_.template get<std::string>("preconditioner.type", "");
            if (precType != "paroverilu0" && precType != "ParOverILU0") {
                invalidateRows_(transposed, *overlapRows_);
            }
            for (const int row : *overlapRows_) {
                rhs[row] = 0.0;
            }

            using ParOperator =
                Opm::GhostLastMatrixAdapter<Matrix, Vector, Vector,
                                            Dune::OwnerOverlapCopyCommunication<int, int>>;
            ParOperator op(transposed, *comm_);
            Dune::FlexibleSolver<ParOperator> solver(op, *comm_, prm_,
                                                     weightsCalculator,
                                                     pressureIndex_);
            solver.apply(solution, rhs, result);
        } else
#endif
        {
            using Operator = Dune::MatrixAdapter<Matrix, Vector, Vector>;
            Operator op(transposed);
            Dune::FlexibleSolver<Operator> solver(op, prm_, weightsCalculator,
                                                  pressureIndex_);
            solver.apply(solution, rhs, result);
        }

        if (!result.converged) {
            OPM_THROW(std::runtime_error,
                      fmt::format("Adjoint linear solver '{}' did not "
                                  "converge in {} iterations "
                                  "(reduction achieved {:.3e})",
                                  spec_, result.iterations,
                                  result.reduction));
        }
        totalIterations_ += result.iterations;
        OpmLog::debug(fmt::format("adjoint linear solve ({}): {} iterations, "
                                  "reduction {:.3e}",
                                  spec_, result.iterations, result.reduction));
    }

    static void invalidateRows_(Matrix& matrix, const std::vector<int>& rows)
    {
        typename Matrix::block_type identity(0.0);
        for (int eq = 0; eq < static_cast<int>(Matrix::block_type::rows); ++eq) {
            identity[eq][eq] = 1.0;
        }
        for (const int row : rows) {
            matrix[row] = 0.0;
            matrix[row][row] = identity;
        }
    }

    std::string spec_{"umfpack"};
    PropertyTree prm_{};
    bool usesCprWeights_ = false;
    bool transposedWeights_ = false;
    std::size_t pressureIndex_ = 1;
#if HAVE_MPI
    const Dune::OwnerOverlapCopyCommunication<int, int>* comm_ = nullptr;
    const std::vector<int>* overlapRows_ = nullptr;
#endif
    int totalIterations_ = 0;
    int numSolves_ = 0;
};

} // namespace Opm

#endif // OPM_ADJOINT_LINEAR_SOLVER_HPP
