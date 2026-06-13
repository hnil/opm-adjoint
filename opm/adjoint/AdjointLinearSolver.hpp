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
 * Parallel runs (serial keeps the FlexibleSolver path above): the
 * forward simulation assembles complete ROWS for interior cells, but the
 * adjoint needs complete COLUMNS, and a plain local transpose drops the
 * cross-rank couplings that a column needs (verified: gradients wrong for
 * np>=3). The parallel path therefore uses TransposedParallelOperator,
 * which applies the EXACT transposed action by scattering A^T over the
 * forward matrix's complete interior rows and accumulating the overlap
 * contributions onto their owners (addOwnerCopyToOwnerCopy) - i.e. it
 * communicates the missing part rather than relying on the local
 * transpose. Preconditioner = opm's parallel ILU0 on the (approximate)
 * local transpose; solver = BiCGSTAB. The preconditioner choice in the
 * spec only sets tol/maxiter in parallel. UMFPACK stays single-rank
 * (parallel falls back to ilu0).
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
#include <dune/istl/solvers.hh>
#if HAVE_MPI
#include <opm/simulators/linalg/ParallelOverlappingILU0.hpp>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/schwarz.hh>
#endif

#include <fmt/format.h>

#include <functional>
#include <stdexcept>
#include <string>

namespace Opm {

#if HAVE_MPI
//! \brief Exact parallel transposed operator y = A^T x.
//!
//! The forward simulation assembles complete *rows* for interior cells,
//! but the adjoint needs complete *columns* — and the local matrix's
//! overlap (ghost) rows do not carry the off-diagonal entries that
//! complete an interior column. A plain local transpose therefore loses
//! the cross-rank couplings A_{ki} where k is interior on another rank
//! (verified: wrong gradients for np>=3).
//!
//! This operator avoids the lossy transpose: it scatters A^T over the
//! complete interior rows of the FORWARD matrix
//!   y[i] += A[k][i]^T x[k]   for interior rows k, all columns i,
//! which writes partial sums into both interior and overlap entries of
//! y, then accumulates the overlap contributions onto their owners with
//! addOwnerCopyToOwnerCopy. The owner of each interior cell then holds
//! the full column sum. The (lossy) local transpose is kept only as the
//! preconditioner matrix (getmat) — an approximation is fine there since
//! the operator action is exact.
template<class Matrix, class Vector, class Comm>
class TransposedParallelOperator
    : public Dune::AssembledLinearOperator<Matrix, Vector, Vector>
{
public:
    using field_type = typename Vector::field_type;

    TransposedParallelOperator(const Matrix& forward,
                               const Matrix& localTranspose,
                               const Comm& comm)
        : forward_(forward)
        , transpose_(localTranspose)
        , comm_(comm)
        , interiorSize_(computeInteriorSize_(comm))
    {}

    Dune::SolverCategory::Category category() const override
    { return Dune::SolverCategory::overlapping; }

    void apply(const Vector& x, Vector& y) const override
    {
        y = 0;
        scatterTransposed_(x, y);
        // accumulate the cross-rank column contributions onto owners
        // (essential: without it the gradients are wrong for np>=3 -
        // the interior-row scatter alone gives an incomplete column)
        comm_.addOwnerCopyToOwnerCopy(y, y);
        // restrict to interior (ghost rows are owned elsewhere)
        for (std::size_t i = interiorSize_; i < y.size(); ++i) {
            y[i] = 0;
        }
    }

    void applyscaleadd(field_type alpha, const Vector& x, Vector& y) const override
    {
        Vector tmp(y);
        apply(x, tmp);
        y.axpy(alpha, tmp);
    }

    const Matrix& getmat() const override
    { return transpose_; }

private:
    void scatterTransposed_(const Vector& x, Vector& y) const
    {
        for (auto row = forward_.begin(); row.index() < interiorSize_; ++row) {
            const auto k = row.index();
            for (auto col = (*row).begin(); col != (*row).end(); ++col) {
                // y[col] += A[k][col]^T x[k]
                (*col).umtv(x[k], y[col.index()]);
            }
        }
    }

    static std::size_t computeInteriorSize_(const Comm& comm)
    {
        std::size_t is = 0;
        const auto& indexSet = comm.indexSet();
        for (auto idx = indexSet.begin(); idx != indexSet.end(); ++idx) {
            if (idx->local().attribute() == 1) { // owner
                is = std::max<std::size_t>(is, idx->local().local());
            }
        }
        return is + 1;
    }

    const Matrix& forward_;
    const Matrix& transpose_;
    const Comm& comm_;
    std::size_t interiorSize_;
};
#endif // HAVE_MPI

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
                      "direct solver; parallel adjoint runs use an iterative "
                      "solver (pass any non-umfpack value, e.g. ilu0)");
        }
        // In parallel the operator is the exact transposed action and the
        // preconditioner is always a block-Jacobi(ILU0) over ranks built
        // from the local transpose (see solveParallel_); the requested
        // preconditioner (cpr/cprt/...) only sets tol/maxiter there.
        if (comm_ && spec_ != "ilu0") {
            OpmLog::info("Parallel adjoint solve uses BiCGSTAB + "
                         "block-ILU0 on the exact transposed operator "
                         "(requested '" + spec_ + "' affects only "
                         "tolerance/iteration limits in parallel).");
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
        Dune::InverseOperatorResult result;
#if HAVE_MPI
        if (comm_) {
            solveParallel_(matrix, rhs, solution, result);
        } else
#endif
        {
            solveSerial_(matrix, rhs, solution, result);
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

    //! \brief Serial iterative solve via FlexibleSolver on the explicit
    //!        local transpose (supports cpr/cprt/ilu0/json).
    void solveSerial_(const Matrix& matrix, Vector& rhs, Vector& solution,
                      Dune::InverseOperatorResult& result)
    {
        Matrix transposed = transposeBlockMatrixSameType(matrix);

        std::function<Vector()> weightsCalculator;
        if (usesCprWeights_) {
            const auto& matrixRef = transposed;
            const bool transposeFlag = transposedWeights_;
            const std::size_t pIdx = pressureIndex_;
            weightsCalculator = [&matrixRef, transposeFlag, pIdx]() {
                return Amg::getQuasiImpesWeights<Matrix, Vector>(
                    matrixRef, pIdx, transposeFlag,
                    /*enable_thread_parallel=*/false);
            };
        }
        using Operator = Dune::MatrixAdapter<Matrix, Vector, Vector>;
        Operator op(transposed);
        Dune::FlexibleSolver<Operator> solver(op, prm_, weightsCalculator,
                                              pressureIndex_);
        solver.apply(solution, rhs, result);
    }

#if HAVE_MPI
    //! \brief Parallel iterative solve. The operator is the EXACT
    //!        transposed action (TransposedParallelOperator); the
    //!        preconditioner is a block-Jacobi(ILU0) over ranks built
    //!        from the local transpose (approximate is fine since the
    //!        operator is exact). Hand-assembled with Dune primitives to
    //!        avoid the PreconditionerFactory, which is pre-instantiated
    //!        only for opm's own operator types.
    void solveParallel_(const Matrix& matrix, Vector& rhs, Vector& solution,
                        Dune::InverseOperatorResult& result)
    {
        using Comm = Dune::OwnerOverlapCopyCommunication<int, int>;

        Matrix transposed = transposeBlockMatrixSameType(matrix);
        invalidateRows_(transposed, *overlapRows_);  // overlap rows -> identity
        for (const int row : *overlapRows_) {
            rhs[row] = 0.0;
        }

        TransposedParallelOperator<Matrix, Vector, Comm> op(matrix, transposed,
                                                            *comm_);

        // opm's parallel ILU0 on the local transpose (approximate: its
        // overlap rows are incomplete, fine for a preconditioner since
        // the operator action above is exact).
        ParallelOverlappingILU0<Matrix, Vector, Vector, Comm>
            prec(transposed, *comm_, /*n=*/0, /*w=*/1.0, MILU_VARIANT::ILU);
        Dune::OverlappingSchwarzScalarProduct<Vector, Comm> sp(*comm_);

        const int verbosity = (comm_->communicator().rank() == 0)
            ? prm_.template get<int>("verbosity", 0) : 0;
        const double reduction = prm_.template get<double>("tol", 1e-14);
        const int maxiter = prm_.template get<int>("maxiter", 300);
        Dune::BiCGSTABSolver<Vector> solver(op, sp, prec, reduction, maxiter,
                                            verbosity);
        solver.apply(solution, rhs, result);
    }
#endif

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
