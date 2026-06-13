# -*- mode: cmake; tab-width: 2; indent-tabs-mode: nil; -*-

###########################################################################
#                                                                         #
# Note: The bulk of the build system is located in the cmake/ directory   #
#       of opm-common. This file only contains the specializations for    #
#       this particular project (modeled on current opm-simulators).      #
#                                                                         #
#       dune.module                Name and version number                #
#       CMakeLists_files.cmake     Path of source files                   #
#       opm-adjoint-prereqs.cmake  Dependencies                           #
#                                                                         #
###########################################################################

cmake_minimum_required(VERSION 3.26)

project(opm-adjoint C CXX)

option(SIBLING_SEARCH "Search for other modules in sibling directories?" ON)

# Sibling-directory guess for opm-common (copied from opm-simulators).
if(SIBLING_SEARCH AND NOT opm-common_DIR)
  get_filename_component(_leaf_dir_name ${PROJECT_BINARY_DIR} NAME)
  get_filename_component(_parent_full_dir ${PROJECT_BINARY_DIR} DIRECTORY)
  get_filename_component(_parent_dir_name ${_parent_full_dir} NAME)
  get_filename_component(_modules_dir ${_parent_full_dir} DIRECTORY)
  if(IS_DIRECTORY ${_modules_dir}/opm-common/${_leaf_dir_name})
    set(opm-common_DIR ${_modules_dir}/opm-common/${_leaf_dir_name})
  else()
    string(REPLACE ${PROJECT_NAME} opm-common _opm_common_leaf ${_leaf_dir_name})
    if(NOT _leaf_dir_name STREQUAL _opm_common_leaf
        AND IS_DIRECTORY ${_parent_full_dir}/${_opm_common_leaf})
      set(opm-common_DIR ${_parent_full_dir}/${_opm_common_leaf})
    elseif(IS_DIRECTORY ${_parent_full_dir}/opm-common)
      set(opm-common_DIR "${_parent_full_dir}/opm-common")
    endif()
  endif()
endif()
if(opm-common_DIR AND NOT IS_DIRECTORY ${opm-common_DIR})
  message(WARNING "Value ${opm-common_DIR} passed to variable"
    " opm-common_DIR is not a directory")
endif()

find_package(opm-common REQUIRED)

# project information is in dune.module. Read this file and set variables.
include (OpmInit)

# all setup common to the OPM library modules is done here
# (includes ${project}-prereqs and CMakeLists_files.cmake).
include (OpmLibMain)

target_link_libraries(opmadjoint
  PUBLIC
    opmsimulators
)

# Match the compile environment of the opm-simulators build: HYPRE's
# imported target carries HAVE_HYPRE=1 on its interface, which
# opm-simulators links privately; mismatching definitions between the
# library and our header-instantiated code must be avoided.
if(USE_HYPRE AND USE_MPI)
  find_package(HYPRE)
  if(HYPRE_FOUND)
    target_link_libraries(opmadjoint PUBLIC HYPRE::HYPRE)
  endif()
endif()

# moduleVersion.cpp provides the version strings the flow banner needs;
# compiled here against this module's generated project-version.h
# (same pattern as opm-flowgeomechanics).
add_library(moduleVersionAdjoint OBJECT opm/simulators/utils/moduleVersion.cpp)
set_property(TARGET moduleVersionAdjoint PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(moduleVersionAdjoint PRIVATE "${PROJECT_BINARY_DIR}")
# for opm/simulators/utils/moduleVersion.hpp and its dependencies
target_link_libraries(moduleVersionAdjoint PRIVATE opmsimulators)
add_dependencies(moduleVersionAdjoint opmadjoint)

add_executable(flow_adjoint
  examples/flow_adjoint.cpp
)
target_link_libraries(flow_adjoint
  PUBLIC
    opmadjoint
  PRIVATE
    moduleVersionAdjoint
)

# Look for the opm-tests repository; sets HAVE_OPM_TESTS / OPM_TESTS_ROOT.
include(Findopm-tests)

# Script-driven verification tests (see adjoint_testing.md). The FD
# tests run many small forward simulations; all of them use the tiny
# 3-cell MODEL_1D_DEBUG deck from opm-tests (branch with adjoint_tests/).
if(HAVE_OPM_TESTS)
  set(_m1d ${OPM_TESTS_ROOT}/adjoint_tests/MODEL_1D_DEBUG/inputfiles)
  set(_testout ${PROJECT_BINARY_DIR}/script-tests)
  if(EXISTS ${_m1d})
    add_test(NAME adjoint_replay_m1d
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-replay-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${_m1d}/MODEL_1D_DEBUG.DATA ${_testout}/replay_m1d)
    add_test(NAME adjoint_gradient_regression
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-gradient-regression.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d} ${_testout}/gradreg)
    add_test(NAME adjoint_fd_pv
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d} MODEL_1D_DEBUG
              ${_testout}/fd_pv)
    add_test(NAME adjoint_fd_trans
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-trans-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d} MODEL_1D_DEBUG
              ${_testout}/fd_trans)
    add_test(NAME adjoint_fd_perm
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-perm-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d} MODEL_1D_DEBUG
              ${_testout}/fd_perm)
    add_test(NAME adjoint_fd_wellctrl
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-ctrl-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d} MODEL_1D_DEBUG
              ${_testout}/fd_ctrl)
    add_test(NAME adjoint_fd_endpoint_swl
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-endpoint-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d}
              ${_testout}/fd_endpoint_swl)
    add_test(NAME adjoint_solver_m1d
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-solver-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d}/MODEL_1D_DEBUG.DATA
              ${_testout}/solver_m1d)
  endif()
  if(EXISTS ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA)
    add_test(NAME adjoint_replay_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-replay-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/replay_spe1)
    add_test(NAME adjoint_matchref_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-matchref-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/matchref)
    # SPE1 needs ~125 backward solves; iterative agreement vs umfpack is
    # bounded by the accumulated lambda error (reduction 1e-14 -> ~1e-6).
    add_test(NAME adjoint_solver_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-solver-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/solver_spe1
              1e-5 cprt)
    # Parallel (np=2) vs serial gradient agreement; the script skips
    # itself when no MPI launcher is available.
    add_test(NAME adjoint_mpi_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-mpi-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/mpi_spe1 2)
  endif()
endif()
# Match the opm build conventions (notably -UNDEBUG in Release when
# WITH_NDEBUG=OFF; an NDEBUG mismatch against the opm-simulators library
# is an ABI break in classes with assertion-only members).
include(UseOptimization)
use_additional_optimization(TARGET flow_adjoint)
