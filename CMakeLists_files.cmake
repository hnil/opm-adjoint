# This file sets up the lists of sources, headers and tests of the
# opm-adjoint module; see opm-flowgeomechanics for the conventions.

list (APPEND MAIN_SOURCE_FILES
  opm/adjoint/AdjointParameters.cpp
)

list (APPEND TEST_SOURCE_FILES
  tests/test_AdjointStorage.cpp
  tests/test_transposeMatrix.cpp
)

list (APPEND TEST_DATA_FILES
)

list (APPEND EXAMPLE_SOURCE_FILES
)

list (APPEND PROGRAM_SOURCE_FILES
)

list (APPEND PUBLIC_HEADER_FILES
  opm/adjoint/AdjointFlowProblem.hpp
  opm/adjoint/AdjointLinearSolver.hpp
  opm/adjoint/AdjointMeta.hpp
  opm/adjoint/AdjointParameters.hpp
  opm/adjoint/AdjointRecorder.hpp
  opm/adjoint/AdjointReplay.hpp
  opm/adjoint/AdjointStorage.hpp
  opm/adjoint/AdjointSystemIO.hpp
  opm/adjoint/AdjointWellModel.hpp
  opm/adjoint/transposeMatrix.hpp
)
