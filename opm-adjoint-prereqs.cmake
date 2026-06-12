# defines that must be present in config.h for our headers
set (opm-adjoint_CONFIG_VAR
  HAVE_OPM_GRID
  HAVE_PTHREAD
  HAVE_MPI
  HAVE_SUITESPARSE_UMFPACK_H
  HAVE_DUNE_ISTL
  DUNE_ISTL_VERSION_MAJOR
  DUNE_ISTL_VERSION_MINOR
  DUNE_ISTL_VERSION_REVISION
  HAVE_SUITESPARSE_UMFPACK
  HAVE_HDF5
  )

# dependencies (modern style: plain find_package calls, cf. opm-simulators)
find_package(Boost COMPONENTS date_time unit_test_framework REQUIRED)
find_package(dune-common REQUIRED)
find_package(dune-istl REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(SuiteSparse COMPONENTS UMFPACK REQUIRED)
find_package(opm-grid REQUIRED)
find_package(opm-simulators REQUIRED)

if(USE_MPI)
  set(HDF5_PREFER_PARALLEL TRUE)
endif()
find_package(HDF5)
if(MPI_FOUND AND HDF5_FOUND AND NOT HDF5_IS_PARALLEL)
  message(WARNING "Parallel OPM flow needs a parallel hdf5; found a serial "
    "one. Continuing without hdf5 file support; the adjoint archive falls "
    "back to the directory store.")
  set(HDF5_FOUND OFF)
  unset(HAVE_HDF5)
endif()

# When using an opm-simulators from the same (super)build or an installed
# one, pull in the transitive dependencies its interface may reference.
if(TARGET opmsimulators)
  get_property(opm-simulators_libs TARGET opmsimulators PROPERTY INTERFACE_LINK_LIBRARIES)
  if(opm-simulators_libs MATCHES ParMETIS::ParMETIS)
    find_package(ParMETIS REQUIRED)
  endif()
  if(opm-simulators_libs MATCHES fmt::fmt)
    find_package(fmt)
  endif()
endif()
