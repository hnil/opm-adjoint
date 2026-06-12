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

#include <opm/common/utility/FileSystem.hpp>

#include <opm/adjoint/AdjointMeta.hpp>
#include <opm/adjoint/AdjointStorage.hpp>
#include <opm/adjoint/AdjointSystemIO.hpp>

#define BOOST_TEST_MODULE AdjointStorageTest
#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#include <dune/common/parallel/mpihelper.hh>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrixindexset.hh>

#include <filesystem>

namespace {

class TempDir
{
public:
    TempDir()
        : path_(std::filesystem::temp_directory_path() /
                Opm::unique_path("adjoint_storage_test%%%%%"))
    {
        std::filesystem::create_directories(path_);
    }

    ~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    std::string str() const
    { return path_.string(); }

private:
    std::filesystem::path path_;
};

using TestMatrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 3, 3>>;
using TestVector = Dune::BlockVector<Dune::FieldVector<double, 3>>;

TestMatrix makeTestMatrix()
{
    constexpr int n = 4;
    Dune::MatrixIndexSet indices(n, n);
    for (int i = 0; i < n; ++i) {
        indices.add(i, i);
        if (i > 0) {
            indices.add(i, i - 1);
        }
        if (i < n - 1) {
            indices.add(i, i + 1);
        }
    }
    TestMatrix matrix;
    indices.exportIdx(matrix);

    double value = 1.0;
    for (auto row = matrix.begin(); row != matrix.end(); ++row) {
        for (auto entry = row->begin(); entry != row->end(); ++entry) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    (*entry)[i][j] = value;
                    value += 0.25;
                }
            }
        }
    }
    return matrix;
}

TestVector makeTestVector()
{
    TestVector vector(5);
    double value = -2.0;
    for (std::size_t i = 0; i < vector.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            vector[i][j] = value;
            value += 0.5;
        }
    }
    return vector;
}

Opm::AdjointMeta makeTestMeta()
{
    Opm::AdjointMeta meta;
    meta.systemSaved = true;
    meta.storageCacheEnabled = false;
    meta.caseName = "TESTCASE";
    for (int k = 0; k < 3; ++k) {
        Opm::AdjointSubstepMeta sub;
        sub.globalIdx = k;
        sub.reportStep = k / 2;
        sub.substepInReport = k % 2;
        sub.startTime = 86400.0 * k;
        sub.dt = 86400.0;
        sub.newtonIterations = 3 + k;
        meta.substeps.push_back(sub);
    }
    return meta;
}

} // anonymous namespace

BOOST_AUTO_TEST_CASE(MatrixDumpRoundtrip)
{
    const auto matrix = makeTestMatrix();
    const auto dump = Opm::AdjointMatrixDump::from(matrix);

    BOOST_CHECK_EQUAL(dump.rows, 4U);
    BOOST_CHECK_EQUAL(dump.blockRows, 3U);
    BOOST_CHECK_EQUAL(dump.blockCols, 3U);
    BOOST_CHECK_EQUAL(dump.columnIndices.size(), matrix.nonzeroes());
    BOOST_CHECK_EQUAL(dump.values.size(), matrix.nonzeroes() * 9U);

    const auto result = Opm::compare(dump, dump);
    BOOST_CHECK(result.structureMatches);
    BOOST_CHECK_EQUAL(result.maxAbsDiff, 0.0);
    BOOST_CHECK_EQUAL(result.maxRelDiff, 0.0);

    auto perturbed = dump;
    perturbed.values[7] += 1e-8;
    const auto diff = Opm::compare(dump, perturbed);
    BOOST_CHECK(diff.structureMatches);
    BOOST_CHECK(diff.maxAbsDiff > 0.0);
    BOOST_CHECK(!diff.withinTolerance(1e-12));

    auto reshaped = dump;
    reshaped.columnIndices[0] += 1;
    BOOST_CHECK(!Opm::compare(dump, reshaped).structureMatches);
}

BOOST_AUTO_TEST_CASE(VectorDumpRoundtrip)
{
    const auto vector = makeTestVector();
    const auto dump = Opm::AdjointVectorDump::from(vector);

    BOOST_CHECK_EQUAL(dump.blockSize, 3U);
    BOOST_CHECK_EQUAL(dump.values.size(), vector.size() * 3U);

    const auto result = Opm::compare(dump, dump);
    BOOST_CHECK(result.structureMatches);
    BOOST_CHECK_EQUAL(result.maxRelDiff, 0.0);
}

BOOST_AUTO_TEST_CASE(RawDirArchiveRoundtrip)
{
    TempDir dir;

    auto meta = makeTestMeta();
    auto matrixDump = Opm::AdjointMatrixDump::from(makeTestMatrix());
    auto vectorDump = Opm::AdjointVectorDump::from(makeTestVector());

    {
        Opm::AdjointArchive archive(dir.str(), Opm::AdjointArchive::Mode::Write);
        BOOST_CHECK(!archive.usesHdf5());
        archive.write(meta, Opm::AdjointGroups::meta(), "meta");
        archive.write(matrixDump, Opm::AdjointGroups::substep(0) + "/system", "jacobian");
        archive.write(vectorDump, Opm::AdjointGroups::substep(0) + "/system", "residual");
        archive.writeText(Opm::AdjointGroups::meta(), "parameters", "test=1\n");
    }

    {
        Opm::AdjointArchive archive(dir.str(), Opm::AdjointArchive::Mode::Read);

        Opm::AdjointMeta metaIn;
        archive.read(metaIn, Opm::AdjointGroups::meta(), "meta");
        BOOST_CHECK(metaIn == meta);

        Opm::AdjointMatrixDump matrixIn;
        archive.read(matrixIn, Opm::AdjointGroups::substep(0) + "/system", "jacobian");
        const auto matrixResult = Opm::compare(matrixDump, matrixIn);
        BOOST_CHECK(matrixResult.structureMatches);
        BOOST_CHECK_EQUAL(matrixResult.maxAbsDiff, 0.0);

        Opm::AdjointVectorDump vectorIn;
        archive.read(vectorIn, Opm::AdjointGroups::substep(0) + "/system", "residual");
        const auto vectorResult = Opm::compare(vectorDump, vectorIn);
        BOOST_CHECK(vectorResult.structureMatches);
        BOOST_CHECK_EQUAL(vectorResult.maxAbsDiff, 0.0);
    }
}

BOOST_AUTO_TEST_CASE(MissingArchiveThrows)
{
    BOOST_CHECK_THROW(Opm::AdjointArchive("/nonexistent/adjoint/archive",
                                          Opm::AdjointArchive::Mode::Read),
                      std::runtime_error);
}

#if HAVE_HDF5
BOOST_AUTO_TEST_CASE(Hdf5ArchiveRoundtrip)
{
    TempDir dir;
    const std::string path = dir.str() + "/archive.h5";
    BOOST_CHECK(Opm::AdjointArchive::isHdf5Path(path));

    auto meta = makeTestMeta();
    auto matrixDump = Opm::AdjointMatrixDump::from(makeTestMatrix());
    auto vectorDump = Opm::AdjointVectorDump::from(makeTestVector());

    {
        Opm::AdjointArchive archive(path, Opm::AdjointArchive::Mode::Write);
        BOOST_CHECK(archive.usesHdf5());
        archive.write(meta, Opm::AdjointGroups::meta(), "meta");
        archive.write(matrixDump, Opm::AdjointGroups::substep(0) + "/system", "jacobian");
        archive.write(vectorDump, Opm::AdjointGroups::substep(0) + "/system", "residual");
        archive.writeText(Opm::AdjointGroups::meta(), "parameters", "test=1\n");
    }

    {
        Opm::AdjointArchive archive(path, Opm::AdjointArchive::Mode::Read);

        Opm::AdjointMeta metaIn;
        archive.read(metaIn, Opm::AdjointGroups::meta(), "meta");
        BOOST_CHECK(metaIn == meta);

        Opm::AdjointMatrixDump matrixIn;
        archive.read(matrixIn, Opm::AdjointGroups::substep(0) + "/system", "jacobian");
        const auto matrixResult = Opm::compare(matrixDump, matrixIn);
        BOOST_CHECK(matrixResult.structureMatches);
        BOOST_CHECK_EQUAL(matrixResult.maxAbsDiff, 0.0);

        Opm::AdjointVectorDump vectorIn;
        archive.read(vectorIn, Opm::AdjointGroups::substep(0) + "/system", "residual");
        const auto vectorResult = Opm::compare(vectorDump, vectorIn);
        BOOST_CHECK(vectorResult.structureMatches);
        BOOST_CHECK_EQUAL(vectorResult.maxAbsDiff, 0.0);
    }
}
#endif // HAVE_HDF5

bool init_unit_test_func()
{
    return true;
}

int main(int argc, char** argv)
{
    // The HDF5 backend uses the parallel communication object, which
    // requires MPI to be initialized.
    Dune::MPIHelper::instance(argc, argv);
    return boost::unit_test::unit_test_main(&init_unit_test_func, argc, argv);
}
