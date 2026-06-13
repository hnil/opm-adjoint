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
 * \brief Parallel context for adjoint runs: ownership masks, the ISTL
 *        communication object for the transposed solves, and gathering
 *        of cell fields for rank-0 output.
 *
 * Parallel correctness of the transposed solve rests on two properties
 * of the flow setup:
 *  - the TPFA linearizer assembles overlap (ghost) rows locally from
 *    synchronized states, so the entry (A^T)_{ij} = A_{ji} of an
 *    interior row i is available locally for every neighbor j (overlap
 *    width 1 suffices for the face term a_{ji}, which both ranks
 *    compute from the same data);
 *  - wells are never split across ranks, so the Schur complement
 *    entries C^T D^-1 B only couple interior cells of the owning rank
 *    and never appear in ghost rows.
 * The local transpose therefore has correct interior rows, and the
 * ghost-last operator (interior rows only + copyOwnerToAll) gives the
 * right global action.
 */
#ifndef OPM_ADJOINT_PARALLEL_HPP
#define OPM_ADJOINT_PARALLEL_HPP

#include <opm/models/utils/propertysystem.hh>
#include <opm/models/discretization/common/fvbaseproperties.hh>

#if HAVE_MPI
#include <opm/simulators/linalg/ExtractParallelGridInformationToISTL.hpp>
#include <opm/simulators/linalg/ParallelIstlInformation.hpp>
#include <opm/simulators/linalg/findOverlapRowsAndColumns.hpp>
#include <dune/istl/owneroverlapcopy.hh>
#endif

#include <dune/grid/common/mcmgmapper.hh>
#include <dune/grid/common/partitionset.hh>

#include <any>
#include <cstddef>
#include <memory>
#include <numeric>
#include <vector>

namespace Opm {

template<class TypeTag>
class AdjointParallel
{
public:
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using GridView = GetPropType<TypeTag, Properties::GridView>;
    using ElementMapper =
        Dune::MultipleCodimMultipleGeomTypeMapper<GridView>;
#if HAVE_MPI
    using IstlComm = Dune::OwnerOverlapCopyCommunication<int, int>;
#endif

    explicit AdjointParallel(const Simulator& simulator)
        : gridComm_(simulator.vanguard().gridView().comm())
    {
        const auto& gridView = simulator.vanguard().gridView();
        const std::size_t numCells = gridView.size(/*codim=*/0);

        ElementMapper mapper(gridView, Dune::mcmgElementLayout());
        interior_.assign(numCells, true);
        cartesianIdx_.resize(numCells);
        for (std::size_t i = 0; i < numCells; ++i) {
            cartesianIdx_[i] = simulator.vanguard().cartesianIndex(i);
        }

        if (gridComm_.size() <= 1) {
            return;
        }
#if HAVE_MPI
        interior_.assign(numCells, false);
        for (const auto& element :
             elements(gridView, Dune::Partitions::interior)) {
            interior_[mapper.index(element)] = true;
        }

        istlComm_ = std::make_unique<IstlComm>(
            simulator.vanguard().grid().comm());
        extractParallelGridInformationToISTL(simulator.vanguard().grid(),
                                             parallelInformation_);
        const auto* parinfo =
            std::any_cast<ParallelISTLInformation>(&parallelInformation_);
        parinfo->copyValuesTo(istlComm_->indexSet(),
                              istlComm_->remoteIndices(), numCells, 1);

        detail::findOverlapAndInterior(simulator.vanguard().grid(), mapper,
                                       overlapRows_, interiorRows_);
#endif
    }

    bool parallel() const
    { return gridComm_.size() > 1; }

    bool interior(std::size_t cell) const
    { return interior_[cell]; }

    const std::vector<int>& overlapRows() const
    { return overlapRows_; }

    //! \brief Global cell id (cartesian index) of a local cell.
    long cartesianIndex(std::size_t cell) const
    { return cartesianIdx_[cell]; }

    //! \brief True if this rank is responsible for a face between local
    //!        cells \p in and \p out (counts every global face once:
    //!        interior-interior faces are always local responsibility,
    //!        interior-ghost faces belong to the rank owning the cell
    //!        with the smaller global id).
    bool ownsFace(std::size_t in, std::size_t out) const
    {
        if (interior_[in] && interior_[out]) {
            return true;
        }
        const std::size_t inner = interior_[in] ? in : out;
        const std::size_t ghost = interior_[in] ? out : in;
        return cartesianIdx_[inner] < cartesianIdx_[ghost];
    }

    Scalar sum(Scalar value) const
    { return gridComm_.sum(value); }

    double max(double value) const
    { return gridComm_.max(value); }

    int rank() const
    { return gridComm_.rank(); }

    //! \brief Gather (global id, value) pairs of interior cells on rank
    //!        0, sorted by global id. On rank != 0 the result is empty.
    //!        For a serial run this returns the field in local order
    //!        (which coincides with global order for our test decks).
    std::vector<Scalar> gatherCellField(const std::vector<Scalar>& field) const
    {
        std::vector<long> ids;
        std::vector<Scalar> values;
        ids.reserve(field.size());
        values.reserve(field.size());
        for (std::size_t i = 0; i < field.size(); ++i) {
            if (interior_[i]) {
                ids.push_back(cartesianIdx_[i]);
                values.push_back(field[i]);
            }
        }
        if (!parallel()) {
            return values;
        }

        const int localSize = static_cast<int>(ids.size());
        std::vector<int> sizes(gridComm_.size(), 0);
        gridComm_.gather(&localSize, sizes.data(), 1, 0);

        std::vector<int> displ(gridComm_.size() + 1, 0);
        std::partial_sum(sizes.begin(), sizes.end(), displ.begin() + 1);
        std::vector<long> allIds(displ.back());
        std::vector<Scalar> allValues(displ.back());
        gridComm_.gatherv(ids.data(), localSize, allIds.data(),
                          sizes.data(), displ.data(), 0);
        gridComm_.gatherv(values.data(), localSize, allValues.data(),
                          sizes.data(), displ.data(), 0);
        if (rank() != 0) {
            return {};
        }

        std::vector<std::size_t> order(allIds.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&allIds](std::size_t a, std::size_t b)
                  { return allIds[a] < allIds[b]; });
        std::vector<Scalar> result(allValues.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            result[i] = allValues[order[i]];
        }
        return result;
    }

    //! \brief Gather text lines from all ranks onto rank 0 (order: by
    //!        rank, then local order). Returns the input unchanged in
    //!        serial; empty on rank != 0.
    std::vector<std::string> gatherLines(const std::vector<std::string>& lines) const
    {
        if (!parallel()) {
            return lines;
        }
        std::string local;
        for (const auto& line : lines) {
            local += line;
            local += '\n';
        }
        const int localSize = static_cast<int>(local.size());
        std::vector<int> sizes(gridComm_.size(), 0);
        gridComm_.gather(&localSize, sizes.data(), 1, 0);
        std::vector<int> displ(gridComm_.size() + 1, 0);
        std::partial_sum(sizes.begin(), sizes.end(), displ.begin() + 1);
        std::vector<char> all(displ.back());
        gridComm_.gatherv(local.data(), localSize, all.data(),
                          sizes.data(), displ.data(), 0);
        if (rank() != 0) {
            return {};
        }
        std::vector<std::string> result;
        std::string current;
        for (const char c : all) {
            if (c == '\n') {
                result.push_back(current);
                current.clear();
            } else {
                current += c;
            }
        }
        return result;
    }

#if HAVE_MPI
    //! \brief ISTL communication for the parallel transposed solves
    //!        (null in serial runs).
    const IstlComm* istlComm() const
    { return istlComm_.get(); }

    //! \brief Make lambda consistent on ghost cells after a solve.
    template<class Vector>
    void makeConsistent(Vector& x) const
    {
        if (istlComm_) {
            istlComm_->copyOwnerToAll(x, x);
        }
    }
#else
    template<class Vector>
    void makeConsistent(Vector&) const
    {}
#endif

private:
    using GridComm = std::remove_cvref_t<
        decltype(std::declval<GridView>().comm())>;

    GridComm gridComm_;
    std::vector<bool> interior_;
    std::vector<long> cartesianIdx_;
    std::vector<int> overlapRows_;
    std::vector<int> interiorRows_;
#if HAVE_MPI
    std::unique_ptr<IstlComm> istlComm_;
    std::any parallelInformation_;
#endif
};

} // namespace Opm

#endif // OPM_ADJOINT_PARALLEL_HPP
