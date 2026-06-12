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
 * \brief Plain-data dumps of (block) sparse matrices and block vectors,
 *        serializable with the generic serializeOp machinery.
 *
 * Used to store the converged residual/Jacobian of every accepted substep
 * in the adjoint archive (--adjoint-save-system) and to compare them with
 * the systems recreated by the replay driver.
 */
#ifndef OPM_ADJOINT_SYSTEM_IO_HPP
#define OPM_ADJOINT_SYSTEM_IO_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace Opm {

//! \brief POD dump of a Dune::BCRSMatrix with dense blocks.
struct AdjointMatrixDump
{
    std::size_t rows{};       //!< number of block rows
    std::size_t cols{};       //!< number of block columns
    std::size_t blockRows{};  //!< rows per block
    std::size_t blockCols{};  //!< columns per block
    std::vector<std::size_t> rowPointers;    //!< CSR row starts (size rows+1)
    std::vector<std::size_t> columnIndices;  //!< CSR block column indices
    std::vector<double> values;              //!< nnz * blockRows * blockCols entries

    template<class Serializer>
    void serializeOp(Serializer& serializer)
    {
        serializer(rows);
        serializer(cols);
        serializer(blockRows);
        serializer(blockCols);
        serializer(rowPointers);
        serializer(columnIndices);
        serializer(values);
    }

    //! \brief Extract a dump from a Dune::BCRSMatrix-like matrix.
    template<class Matrix>
    static AdjointMatrixDump from(const Matrix& matrix)
    {
        using Block = typename Matrix::block_type;
        AdjointMatrixDump dump;
        dump.rows = matrix.N();
        dump.cols = matrix.M();
        dump.blockRows = Block::rows;
        dump.blockCols = Block::cols;
        dump.rowPointers.reserve(dump.rows + 1);
        dump.rowPointers.push_back(0);
        dump.columnIndices.reserve(matrix.nonzeroes());
        dump.values.reserve(matrix.nonzeroes() * Block::rows * Block::cols);
        for (auto row = matrix.begin(); row != matrix.end(); ++row) {
            for (auto entry = row->begin(); entry != row->end(); ++entry) {
                dump.columnIndices.push_back(entry.index());
                for (std::size_t i = 0; i < Block::rows; ++i) {
                    for (std::size_t j = 0; j < Block::cols; ++j) {
                        dump.values.push_back(static_cast<double>((*entry)[i][j]));
                    }
                }
            }
            dump.rowPointers.push_back(dump.columnIndices.size());
        }
        return dump;
    }
};

//! \brief POD dump of a Dune::BlockVector with FieldVector blocks.
struct AdjointVectorDump
{
    std::size_t blockSize{};       //!< entries per block
    std::vector<double> values;    //!< size() * blockSize entries

    template<class Serializer>
    void serializeOp(Serializer& serializer)
    {
        serializer(blockSize);
        serializer(values);
    }

    template<class Vector>
    static AdjointVectorDump from(const Vector& vector)
    {
        using Block = typename Vector::block_type;
        AdjointVectorDump dump;
        dump.blockSize = Block::dimension;
        dump.values.reserve(vector.size() * Block::dimension);
        for (std::size_t i = 0; i < vector.size(); ++i) {
            for (std::size_t j = 0; j < Block::dimension; ++j) {
                dump.values.push_back(static_cast<double>(vector[i][j]));
            }
        }
        return dump;
    }
};

//! \brief Result of comparing two dumps.
struct AdjointCompareResult
{
    bool structureMatches{true};  //!< same sparsity pattern / sizes
    double maxAbsDiff{0.0};
    double maxRelDiff{0.0};       //!< |a-b| / max(|a|, |b|, floor)
    std::string mismatch{};       //!< description of a structural mismatch

    bool withinTolerance(double relTol) const
    {
        return structureMatches && maxRelDiff <= relTol;
    }
};

namespace detail {

inline void updateDiffs(double a, double b, AdjointCompareResult& result,
                        double floor, double absFloor)
{
    const double absDiff = std::abs(a - b);
    result.maxAbsDiff = std::max(result.maxAbsDiff, absDiff);
    if (absDiff <= absFloor) {
        // Treated as rounding noise: does not enter the relative criterion.
        return;
    }
    const double scale = std::max({std::abs(a), std::abs(b), floor});
    result.maxRelDiff = std::max(result.maxRelDiff, absDiff / scale);
}

} // namespace detail

//! \brief Compare two vector dumps entry by entry.
//! \param floor Magnitudes below this only enter the relative diff through
//!              the absolute difference (avoids 0-vs-1e-300 blowups).
//! \param absFloor Entries whose absolute difference is below this are
//!                 counted as rounding noise and excluded from the relative
//!                 criterion (they still show up in maxAbsDiff).
inline AdjointCompareResult compare(const AdjointVectorDump& a,
                                    const AdjointVectorDump& b,
                                    double floor = 1e-300,
                                    double absFloor = 0.0)
{
    AdjointCompareResult result;
    if (a.blockSize != b.blockSize || a.values.size() != b.values.size()) {
        result.structureMatches = false;
        result.mismatch = "vector size/block mismatch: " +
            std::to_string(a.values.size()) + "/" + std::to_string(a.blockSize) +
            " vs " + std::to_string(b.values.size()) + "/" + std::to_string(b.blockSize);
        return result;
    }
    for (std::size_t i = 0; i < a.values.size(); ++i) {
        detail::updateDiffs(a.values[i], b.values[i], result, floor, absFloor);
    }
    return result;
}

//! \brief Compare two matrix dumps; requires identical sparsity structure.
inline AdjointCompareResult compare(const AdjointMatrixDump& a,
                                    const AdjointMatrixDump& b,
                                    double floor = 1e-300,
                                    double absFloor = 0.0)
{
    AdjointCompareResult result;
    if (a.rows != b.rows || a.cols != b.cols ||
        a.blockRows != b.blockRows || a.blockCols != b.blockCols ||
        a.rowPointers != b.rowPointers || a.columnIndices != b.columnIndices) {
        result.structureMatches = false;
        result.mismatch = "matrix structure mismatch (rows " +
            std::to_string(a.rows) + " vs " + std::to_string(b.rows) +
            ", nnz " + std::to_string(a.columnIndices.size()) +
            " vs " + std::to_string(b.columnIndices.size()) + ")";
        return result;
    }
    for (std::size_t i = 0; i < a.values.size(); ++i) {
        detail::updateDiffs(a.values[i], b.values[i], result, floor, absFloor);
    }
    return result;
}

} // namespace Opm

#endif // OPM_ADJOINT_SYSTEM_IO_HPP
