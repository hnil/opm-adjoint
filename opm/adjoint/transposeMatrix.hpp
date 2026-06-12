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
 * \brief Explicit transposition of block-sparse (BCRS) matrices.
 *
 * Used by the adjoint solver: the adjoint of one timestep solves
 * J^T lambda = rhs with the same Jacobian J the forward simulation
 * assembled. Building J^T explicitly costs one structural pass plus a
 * value copy - negligible next to the solve - and makes every existing
 * solver and preconditioner usable unchanged.
 */
#ifndef OPM_TRANSPOSE_MATRIX_HPP
#define OPM_TRANSPOSE_MATRIX_HPP

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixindexset.hh>

#include <cstddef>

namespace Opm {

//! \brief Transpose a dense block in place semantics: returns the
//!        transposed block. Requires a square block type.
template<class Block>
Block transposeBlock(const Block& block)
{
    static_assert(static_cast<int>(Block::rows) == static_cast<int>(Block::cols),
                  "transposeBlock requires square blocks");
    Block transposed;
    for (std::size_t i = 0; i < Block::rows; ++i) {
        for (std::size_t j = 0; j < Block::cols; ++j) {
            transposed[i][j] = block[j][i];
        }
    }
    return transposed;
}

//! \brief Build the transpose of a BCRS matrix with square dense blocks.
//!
//! The returned matrix has the transposed sparsity pattern and each block
//! is the transpose of the corresponding source block:
//! (A^T)_{ji} = (A_{ij})^T.
template<class Matrix>
Matrix transposeBlockMatrix(const Matrix& matrix)
{
    Dune::MatrixIndexSet indices(matrix.M(), matrix.N());
    for (auto row = matrix.begin(); row != matrix.end(); ++row) {
        for (auto entry = row->begin(); entry != row->end(); ++entry) {
            indices.add(entry.index(), row.index());
        }
    }

    Matrix transposed;
    indices.exportIdx(transposed);

    for (auto row = matrix.begin(); row != matrix.end(); ++row) {
        for (auto entry = row->begin(); entry != row->end(); ++entry) {
            transposed[entry.index()][row.index()] = transposeBlock(*entry);
        }
    }
    return transposed;
}

} // namespace Opm

#endif // OPM_TRANSPOSE_MATRIX_HPP
