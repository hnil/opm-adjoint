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
