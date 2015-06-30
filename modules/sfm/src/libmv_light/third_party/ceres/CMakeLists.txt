# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
# http://code.google.com/p/ceres-solver/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: keir@google.com (Keir Mierle)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

IF (COMMAND cmake_policy)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF (COMMAND cmake_policy)

PROJECT(CERES C CXX)

# Set up the git hook to make Gerrit Change-Id: lines in commit messages.
SET (LOCAL_GIT_DIRECTORY)
IF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  # .git directory can be found on Unix based system, or on Windows with
  # Git Bash (shipped with msysgit)
  SET (LOCAL_GIT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
ELSE (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  # TODO(keir) Add proper Windows support
ENDIF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)

IF (EXISTS ${LOCAL_GIT_DIRECTORY})
  IF (NOT EXISTS ${LOCAL_GIT_DIRECTORY}/hooks/commit-msg)
    # Download the hook only if it is not already present
    FILE(DOWNLOAD https://ceres-solver-review.googlesource.com/tools/hooks/commit-msg
         ${CMAKE_CURRENT_BINARY_DIR}/commit-msg)

    # Make the downloaded file executable, since it is not by default.
    FILE(COPY ${CMAKE_CURRENT_BINARY_DIR}/commit-msg
         DESTINATION ${LOCAL_GIT_DIRECTORY}/hooks/
         FILE_PERMISSIONS
           OWNER_READ OWNER_WRITE OWNER_EXECUTE
           GROUP_READ GROUP_WRITE GROUP_EXECUTE
           WORLD_READ WORLD_EXECUTE)
  ENDIF (NOT EXISTS ${LOCAL_GIT_DIRECTORY}/hooks/commit-msg)
ENDIF (EXISTS ${LOCAL_GIT_DIRECTORY})

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Important: Always bump the second number (e.g. 1.3.x to 1.4.0) for any
# release that changes the ABI. The ABI changes for almost any modification to
# include/ceres (e.g. the public API). If you are unsure about whether
# something is an ABI change, please ask on the list.
#
# For versions without ABI changes, bump the smallest number in CERES_VERSION,
# but leave the CERES_ABI_VERSION unchanged.
SET(CERES_VERSION_MAJOR 1)
SET(CERES_VERSION_MINOR 5)
SET(CERES_VERSION_PATCH 0)
SET(CERES_VERSION
    ${CERES_VERSION_MAJOR}.${CERES_VERSION_MINOR}.${CERES_VERSION_PATCH})
SET(CERES_ABI_VERSION 1.5.0)

ENABLE_TESTING()

OPTION(BUILD_TESTING
       "Enable tests"
       OFF)

OPTION(BUILD_ANDROID
       "Build for Android. Use build_android.sh instead of setting this."
       OFF)

# To get a more static build, try the following line on Mac and Linux:
# SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})

# Default locations to search for on various platforms.
LIST(APPEND SEARCH_LIBS /usr/lib)
LIST(APPEND SEARCH_LIBS /usr/local/lib)
LIST(APPEND SEARCH_LIBS /usr/local/homebrew/lib) # Mac OS X
LIST(APPEND SEARCH_LIBS /opt/local/lib)

LIST(APPEND SEARCH_HEADERS /usr/include)
LIST(APPEND SEARCH_HEADERS /usr/local/include)
LIST(APPEND SEARCH_HEADERS /usr/local/homebrew/include) # Mac OS X
LIST(APPEND SEARCH_HEADERS /opt/local/include)

# Locations to search for Eigen
SET(EIGEN_SEARCH_HEADERS ${SEARCH_HEADERS})
LIST(APPEND EIGEN_SEARCH_HEADERS /usr/include/eigen3) # Ubuntu 10.04's default location.
LIST(APPEND EIGEN_SEARCH_HEADERS /usr/local/include/eigen3)
LIST(APPEND EIGEN_SEARCH_HEADERS /usr/local/homebrew/include/eigen3)  # Mac OS X
LIST(APPEND EIGEN_SEARCH_HEADERS /opt/local/var/macports/software/eigen3/opt/local/include/eigen3) # Mac OS X

# Locations to search for SuiteSparse
SET(SUITESPARSE_SEARCH_LIBS ${SEARCH_LIBS})
LIST(APPEND SUITESPARSE_SEARCH_LIBS /usr/lib/suitesparse) # Ubuntu
LIST(APPEND SUITESPARSE_SEARCH_LIBS /usr/local/lib/suitesparse)
LIST(APPEND SUITESPARSE_SEARCH_LIBS /opt/local/lib/ufsparse) # Mac OS X

SET(SUITESPARSE_SEARCH_HEADERS ${SEARCH_HEADERS})
LIST(APPEND SUITESPARSE_SEARCH_HEADERS /usr/include/suitesparse) # Ubuntu
LIST(APPEND SUITESPARSE_SEARCH_HEADERS /usr/local/include/suitesparse)
LIST(APPEND SUITESPARSE_SEARCH_HEADERS /opt/local/include/ufsparse) # Mac OS X

SET(CXSPARSE_SEARCH_LIBS ${SEARCH_LIBS})
SET(CXSPARSE_SEARCH_HEADERS ${SEARCH_HEADERS})
LIST(APPEND CXSPARSE_SEARCH_HEADERS /usr/include/suitesparse) # Ubuntu

IF ((NOT DEFINED SUITESPARSE) OR (DEFINED SUITESPARSE AND SUITESPARSE))
# Check for SuiteSparse dependencies
MESSAGE("-- Check for AMD")
SET(AMD_FOUND TRUE)

FIND_LIBRARY(AMD_LIB NAMES amd PATHS ${SUITESPARSE_SEARCH_LIBS})
IF (EXISTS ${AMD_LIB})
  MESSAGE("-- Found AMD library: ${AMD_LIB}")
ELSE (EXISTS ${AMD_LIB})
  MESSAGE("-- Did not find AMD library")
  SET(AMD_FOUND FALSE)
ENDIF (EXISTS ${AMD_LIB})

FIND_PATH(AMD_INCLUDE NAMES amd.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
IF (EXISTS ${AMD_INCLUDE})
  MESSAGE("-- Found AMD header in: ${AMD_INCLUDE}")
ELSE (EXISTS ${AMD_INCLUDE})
  MESSAGE("-- Did not find AMD header")
  SET(AMD_FOUND FALSE)
ENDIF (EXISTS ${AMD_INCLUDE})

MESSAGE("-- Check for CAMD")
SET(CAMD_FOUND TRUE)

FIND_LIBRARY(CAMD_LIB NAMES camd PATHS ${SUITESPARSE_SEARCH_LIBS})
IF (EXISTS ${CAMD_LIB})
  MESSAGE("-- Found CAMD library: ${CAMD_LIB}")
ELSE (EXISTS ${CAMD_LIB})
  MESSAGE("-- Did not find CAMD library")
  SET(CAMD_FOUND FALSE)
ENDIF (EXISTS ${CAMD_LIB})

FIND_PATH(CAMD_INCLUDE NAMES camd.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
IF (EXISTS ${CAMD_INCLUDE})
  MESSAGE("-- Found CAMD header in: ${CAMD_INCLUDE}")
ELSE (EXISTS ${CAMD_INCLUDE})
  MESSAGE("-- Did not find CAMD header")
  SET(CAMD_FOUND FALSE)
ENDIF (EXISTS ${CAMD_INCLUDE})

MESSAGE("-- Check for COLAMD")
SET(COLAMD_FOUND TRUE)

FIND_LIBRARY(COLAMD_LIB NAMES colamd PATHS ${SUITESPARSE_SEARCH_LIBS})
IF (EXISTS ${COLAMD_LIB})
  MESSAGE("-- Found COLAMD library: ${COLAMD_LIB}")
ELSE (EXISTS ${COLAMD_LIB})
  MESSAGE("-- Did not find COLAMD library")
  SET(COLAMD_FOUND FALSE)
ENDIF (EXISTS ${COLAMD_LIB})

FIND_PATH(COLAMD_INCLUDE NAMES colamd.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
IF (EXISTS ${COLAMD_INCLUDE})
  MESSAGE("-- Found COLAMD header in: ${COLAMD_INCLUDE}")
ELSE (EXISTS ${COLAMD_INCLUDE})
  MESSAGE("-- Did not find COLAMD header")
  SET(COLAMD_FOUND FALSE)
ENDIF (EXISTS ${COLAMD_INCLUDE})

MESSAGE("-- Check for CCOLAMD")
SET(CCOLAMD_FOUND TRUE)

FIND_LIBRARY(CCOLAMD_LIB NAMES ccolamd PATHS ${SUITESPARSE_SEARCH_LIBS})
IF (EXISTS ${CCOLAMD_LIB})
  MESSAGE("-- Found CCOLAMD library: ${CCOLAMD_LIB}")
ELSE (EXISTS ${CCOLAMD_LIB})
  MESSAGE("-- Did not find CCOLAMD library")
  SET(CCOLAMD_FOUND FALSE)
ENDIF (EXISTS ${CCOLAMD_LIB})

FIND_PATH(CCOLAMD_INCLUDE NAMES ccolamd.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
IF (EXISTS ${CCOLAMD_INCLUDE})
  MESSAGE("-- Found CCOLAMD header in: ${CCOLAMD_INCLUDE}")
ELSE (EXISTS ${CCOLAMD_INCLUDE})
  MESSAGE("-- Did not find CCOLAMD header")
  SET(CCOLAMD_FOUND FALSE)
ENDIF (EXISTS ${CCOLAMD_INCLUDE})

MESSAGE("-- Check for CHOLMOD")
SET(CHOLMOD_FOUND TRUE)

FIND_LIBRARY(CHOLMOD_LIB NAMES cholmod PATHS ${SUITESPARSE_SEARCH_LIBS})
IF (EXISTS ${CHOLMOD_LIB})
  MESSAGE("-- Found CHOLMOD library: ${CHOLMOD_LIB}")
ELSE (EXISTS ${CHOLMOD_LIB})
  MESSAGE("-- Did not find CHOLMOD library")
  SET(CHOLMOD_FOUND FALSE)
ENDIF (EXISTS ${CHOLMOD_LIB})

FIND_PATH(CHOLMOD_INCLUDE NAMES cholmod.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
IF (EXISTS ${CHOLMOD_INCLUDE})
  MESSAGE("-- Found CHOLMOD header in: ${CHOLMOD_INCLUDE}")
ELSE (EXISTS ${CHOLMOD_INCLUDE})
  MESSAGE("-- Did not find CHOLMOD header")
  SET(CHOLMOD_FOUND FALSE)
ENDIF (EXISTS ${CHOLMOD_INCLUDE})

# If SuiteSparse version is >= 4 then SuiteSparse_config is required.
# For SuiteSparse 3, UFconfig.h is required.
MESSAGE("-- Check for SuiteSparse_config (SuiteSparse v4)")
SET(SUITESPARSE_CONFIG_FOUND TRUE)

FIND_LIBRARY(SUITESPARSE_CONFIG_LIB
             NAMES suitesparseconfig
             PATHS ${SUITESPARSE_SEARCH_LIBS})
IF (EXISTS ${SUITESPARSE_CONFIG_LIB})
  MESSAGE("-- Found SuiteSparse_config library: ${SUITESPARSE_CONFIG_LIB}")
ELSE (EXISTS ${SUITESPARSE_CONFIG_LIB})
  MESSAGE("-- Did not find SuiteSparse_config library")
ENDIF (EXISTS ${SUITESPARSE_CONFIG_LIB})

FIND_PATH(SUITESPARSE_CONFIG_INCLUDE
          NAMES SuiteSparse_config.h
          PATHS ${SUITESPARSE_SEARCH_HEADERS})
IF (EXISTS ${SUITESPARSE_CONFIG_INCLUDE})
  MESSAGE("-- Found SuiteSparse_config header in: ${SUITESPARSE_CONFIG_INCLUDE}")
ELSE (EXISTS ${SUITESPARSE_CONFIG_INCLUDE})
  MESSAGE("-- Did not find SuiteSparse_config header")
ENDIF (EXISTS ${SUITESPARSE_CONFIG_INCLUDE})

IF (NOT EXISTS ${SUITESPARSE_CONFIG_LIB} OR NOT EXISTS ${SUITESPARSE_CONFIG_INCLUDE})
  SET(SUITESPARSE_CONFIG_FOUND FALSE)
ENDIF (NOT EXISTS ${SUITESPARSE_CONFIG_LIB} OR NOT EXISTS ${SUITESPARSE_CONFIG_INCLUDE})

MESSAGE("-- Check for UFconfig (SuiteSparse v3)")
SET(UFCONFIG_FOUND TRUE)

FIND_PATH(UFCONFIG_INCLUDE
          NAMES UFconfig.h
          PATHS ${SUITESPARSE_SEARCH_HEADERS})
IF (EXISTS ${UFCONFIG_INCLUDE})
  MESSAGE("-- Found UFconfig header in: ${UFCONFIG_INCLUDE}")
ELSE (EXISTS ${UFCONFIG_INCLUDE})
  MESSAGE("-- Did not find UFconfig header")
  SET(UFCONFIG_FOUND FALSE)
ENDIF (EXISTS ${UFCONFIG_INCLUDE})

MESSAGE("-- Check for METIS (optional)")
FIND_LIBRARY(METIS_LIB NAMES metis PATHS ${SUITESPARSE_SEARCH_LIBS})

IF (EXISTS ${METIS_LIB})
  MESSAGE("-- Found METIS library: ${METIS_LIB}")
ELSE (EXISTS ${METIS_LIB})
  MESSAGE("-- Did not find METIS library")
ENDIF (EXISTS ${METIS_LIB})

SET(BLAS_AND_LAPACK_FOUND TRUE)
IF (${APPLE})
  # Mac OS X has LAPACK/BLAS bundled in a framework called
  # "vecLib". Search for that instead of for the normal "lapack"
  # library.
  FIND_LIBRARY(LAPACK_LIB NAMES vecLib)
ELSE (${APPLE})
  FIND_LIBRARY(BLAS_LIB NAMES blas)
  IF (EXISTS ${BLAS_LIB})
    MESSAGE("-- Found BLAS library: ${BLAS_LIB}")
  ELSE (EXISTS ${BLAS_LIB})
    MESSAGE("-- Did not find BLAS library")
    SET(BLAS_AND_LAPACK_FOUND FALSE)
  ENDIF (EXISTS ${BLAS_LIB})
  FIND_LIBRARY(LAPACK_LIB NAMES lapack)
ENDIF (${APPLE})

IF (EXISTS ${LAPACK_LIB})
  MESSAGE("-- Found LAPACK library: ${LAPACK_LIB}")
ELSE (EXISTS ${LAPACK_LIB})
  SET(BLAS_AND_LAPACK_FOUND FALSE)
  MESSAGE("-- Did not find LAPACK library")
ENDIF (EXISTS ${LAPACK_LIB})

SET(SUITESPARSE_FOUND
    ${AMD_FOUND} AND
    ${CAMD_FOUND} AND
    ${COLAMD_FOUND} AND
    ${CCOLAMD_FOUND} AND
    ${CHOLMOD_FOUND} AND
    (${SUITESPARSE_CONFIG_FOUND} OR ${UFCONFIG_FOUND}) AND
    ${BLAS_AND_LAPACK_FOUND})

ENDIF ((NOT DEFINED SUITESPARSE) OR (DEFINED SUITESPARSE AND SUITESPARSE))
# By default, if all of SuiteSparse's dependencies are found, Ceres is
# built with SuiteSparse support. -DSUITESPARSE=ON/OFF can be used to
# enable/disable SuiteSparse explicitly.
IF (DEFINED SUITESPARSE)
  IF (${SUITESPARSE})
    IF (NOT ${SUITESPARSE_FOUND})
      MESSAGE(FATAL_ERROR "One or more of SuiteSparse's dependencies was not found")
    ENDIF (NOT ${SUITESPARSE_FOUND})
  ELSE (${SUITESPARSE})
    ADD_DEFINITIONS(-DCERES_NO_SUITESPARSE)
  ENDIF (${SUITESPARSE})
ELSE (DEFINED SUITESPARSE)
  IF (${SUITESPARSE_FOUND})
    MESSAGE("-- Found all SuiteSparse dependencies. Building with SuiteSparse")
    SET(SUITESPARSE ON)
  ELSE (${SUITESPARSE_FOUND})
    MESSAGE("-- Did not find all SuiteSparse dependencies. Building without SuiteSparse")
    SET(SUITESPARSE OFF)
    ADD_DEFINITIONS(-DCERES_NO_SUITESPARSE)
  ENDIF (${SUITESPARSE_FOUND})
ENDIF (DEFINED SUITESPARSE)

# By default, if all of CXSparse's dependencies are found, Ceres is
# built with CXSparse support. -DCXSPARSE=ON/OFF can be used to
# enable/disable CXSparse explicitly.
IF ((NOT DEFINED CXSPARSE) OR (DEFINED CXSPARSE AND CXSPARSE))
MESSAGE("-- Check for CXSparse")
SET(CXSPARSE_FOUND ON)

FIND_LIBRARY(CXSPARSE_LIB NAMES cxsparse PATHS ${CXSPARSE_SEARCH_LIBS})
IF (EXISTS ${CXSPARSE_LIB})
  MESSAGE("-- Found CXSparse library in: ${CXSPARSE_LIB}")
ELSE (EXISTS ${CXSPARSE_LIB})
  MESSAGE("-- Did not find CXSparse header")
  SET(CXSPARSE_FOUND FALSE)
ENDIF (EXISTS ${CXSPARSE_LIB})

FIND_PATH(CXSPARSE_INCLUDE NAMES cs.h PATHS ${CXSPARSE_SEARCH_HEADERS})
IF (EXISTS ${CXSPARSE_INCLUDE})
  MESSAGE("-- Found CXSparse header in: ${CXSPARSE_INCLUDE}")
ELSE (EXISTS ${CXSPARSE_INCLUDE})
  MESSAGE("-- Did not find CXSparse header")
  SET(CXSPARSE_FOUND FALSE)
ENDIF (EXISTS ${CXSPARSE_INCLUDE})
ENDIF ((NOT DEFINED CXSPARSE) OR (DEFINED CXSPARSE AND CXSPARSE))

IF (DEFINED CXSPARSE)
  IF (${CXSPARSE})
    IF (NOT ${CXSPARSE_FOUND})
      MESSAGE(FATAL_ERROR "-- CXSparse not found.")
    ENDIF (NOT ${CXSPARSE_FOUND})
  ELSE (${CXSPARSE})
    ADD_DEFINITIONS(-DCERES_NO_CXSPARSE)
  ENDIF (${CXSPARSE})
ELSE (DEFINED CXSPARSE)
  IF (${CXSPARSE_FOUND})
    MESSAGE("-- Building with CXSparse support.")
    SET(CXSPARSE ON)
  ELSE (${CXSPARSE_FOUND})
    MESSAGE("-- Building without CXSparse.")
    SET(CXSPARSE OFF)
    ADD_DEFINITIONS(-DCERES_NO_CXSPARSE)
  ENDIF (${CXSPARSE_FOUND})
ENDIF (DEFINED CXSPARSE)

# Google Flags
OPTION(GFLAGS
       "Enable Google Flags."
       ON)

IF (${GFLAGS})
  MESSAGE("-- Check for Google Flags")
  FIND_LIBRARY(GFLAGS_LIB NAMES gflags PATHS ${SEARCH_LIBS})
  IF (NOT GFLAGS_LIB)
    MESSAGE(FATAL_ERROR
            "Can't find Google Flags. Please specify: "
            "-DGFLAGS_LIB=...")
  ENDIF (NOT GFLAGS_LIB)
  MESSAGE("-- Found Google Flags library: ${GFLAGS_LIB}")
  FIND_PATH(GFLAGS_INCLUDE NAMES gflags/gflags.h PATHS ${SEARCH_HEADERS})
  IF (NOT EXISTS ${GFLAGS_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find Google Flags. Please specify: "
            "-DGFLAGS_INCLUDE=...")
  ENDIF (NOT EXISTS ${GFLAGS_INCLUDE})
  MESSAGE("-- Found Google Flags header in: ${GFLAGS_INCLUDE}")
ELSE (${GFLAGS})
  MESSAGE("-- Google Flags disabled; no tests or tools will be built!")
  ADD_DEFINITIONS(-DCERES_NO_GFLAGS)
ENDIF (${GFLAGS})

# Google Logging
IF (NOT ${BUILD_ANDROID})
  MESSAGE("-- Check for Google Log")
  FIND_LIBRARY(GLOG_LIB NAMES glog PATHS ${SEARCH_LIBS})
  IF (NOT GLOG_LIB)
    MESSAGE(FATAL_ERROR
            "Can't find Google Log. Please specify: "
            "-DGLOG_LIB=...")
  ENDIF (NOT GLOG_LIB)
  MESSAGE("-- Found Google Log library: ${GLOG_LIB}")

  FIND_PATH(GLOG_INCLUDE NAMES glog/logging.h PATHS ${SEARCH_HEADERS})
  IF (NOT EXISTS ${GLOG_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find Google Log. Please specify: "
            "-DGLOG_INCLUDE=...")
  ENDIF (NOT EXISTS ${GLOG_INCLUDE})
  MESSAGE("-- Found Google Log header in: ${GLOG_INCLUDE}")
ELSE (NOT ${BUILD_ANDROID})
  SET(GLOG_LIB miniglog)
  MESSAGE("-- Using minimal Glog substitute for Android (library): ${GLOG_LIB}")
  SET(GLOG_INCLUDE internal/ceres/miniglog)
  MESSAGE("-- Using minimal Glog substitute for Android (include): ${GLOG_INCLUDE}")
ENDIF (NOT ${BUILD_ANDROID})

# Eigen
MESSAGE("-- Check for Eigen 3.x")
FIND_PATH(EIGEN_INCLUDE NAMES Eigen/Core PATHS ${EIGEN_SEARCH_HEADERS})
IF (NOT EXISTS ${EIGEN_INCLUDE})
  MESSAGE(FATAL_ERROR "Can't find Eigen. Try passing -DEIGEN_INCLUDE=...")
ENDIF (NOT EXISTS ${EIGEN_INCLUDE})
MESSAGE("-- Found Eigen 3.x: ${EIGEN_INCLUDE}")

# Template specializations for the Schur complement based solvers. If
# compile time, binary size or compiler performance is an issue, you
# may consider disabling this.
OPTION(SCHUR_SPECIALIZATIONS
       "Enable fixed-size schur specializations."
       ON)

IF (NOT ${SCHUR_SPECIALIZATIONS})
  ADD_DEFINITIONS(-DCERES_RESTRICT_SCHUR_SPECIALIZATION)
  MESSAGE("-- Disabling Schur specializations (faster compiles)")
ENDIF (NOT ${SCHUR_SPECIALIZATIONS})

# Line search minimizer is useful for large scale problems or when
# sparse linear algebra libraries are not available. If compile time,
# binary size or compiler performance is an issue, consider disabling
# this.
OPTION(LINE_SEARCH_MINIMIZER
       "Enable the line search minimizer."
       ON)

IF (NOT ${LINE_SEARCH_MINIMIZER})
  ADD_DEFINITIONS(-DCERES_NO_LINE_SEARCH_MINIMIZER)
  MESSAGE("-- Disabling line search minimizer")
ENDIF (NOT ${LINE_SEARCH_MINIMIZER})

OPTION(CUSTOM_BLAS
       "Use handcoded BLAS routines (usually faster) instead of Eigen."
       ON)

IF (NOT ${CUSTOM_BLAS})
  ADD_DEFINITIONS(-DCERES_NO_CUSTOM_BLAS)
  MESSAGE("-- Disabling custom blas")
ENDIF (NOT ${CUSTOM_BLAS})

# Multithreading using OpenMP
OPTION(OPENMP
       "Enable threaded solving in Ceres (requires OpenMP)"
       ON)

IF (${OPENMP})
  FIND_PACKAGE(OpenMP)
ENDIF (${OPENMP})

IF(${OPENMP_FOUND})
  MESSAGE("-- Found OpenMP.")
  ADD_DEFINITIONS(-DCERES_USE_OPENMP)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  IF ("${UNIX}")
    # At least on Linux, we need pthreads to be enabled for mutex to
    # compile.  This may not work on Windows or Android.
    FIND_PACKAGE(Threads REQUIRED)
    SET(STATIC_LIBRARY_FLAGS
        "${STATIC_LIBRARY_FLAGS} ${CMAKE_THREAD_LIBS_INIT}")
    SET(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_THREAD_LIBS_INIT}")
    ADD_DEFINITIONS(-DCERES_HAVE_PTHREAD)
    ADD_DEFINITIONS(-DCERES_HAVE_RWLOCK)
  ENDIF ("${UNIX}")
ELSE(${OPENMP_FOUND})
  MESSAGE("-- Can't find OpenMP. Disabling multithreading.")
  ADD_DEFINITIONS(-DCERES_NO_THREADS)
ENDIF(${OPENMP_FOUND})

# Disable threads in mutex.h. Someday, after there is OpenMP support in
# Android, this can get removed. Also turn on a workaround for an NDK bug.
IF (${BUILD_ANDROID})
  ADD_DEFINITIONS(-DCERES_NO_THREADS)
  ADD_DEFINITIONS(-DCERES_WORK_AROUND_ANDROID_NDK_COMPILER_BUG)
ENDIF (${BUILD_ANDROID})

# Protocol buffers
OPTION(PROTOBUF
       "Enable protocol buffers support."
       ON)

IF (${PROTOBUF})
  FIND_PACKAGE(Protobuf)
  IF (${PROTOBUF_FOUND})
    INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIRS})
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/internal)
  ELSE (${PROTOBUF_FOUND})
    ADD_DEFINITIONS(-DCERES_NO_PROTOCOL_BUFFERS)
  ENDIF (${PROTOBUF_FOUND})
ELSE (${PROTOBUF})
  ADD_DEFINITIONS(-DCERES_NO_PROTOCOL_BUFFERS)
ENDIF (${PROTOBUF})

OPTION(DISABLE_TR1
       "Don't use TR1. This replaces some hash tables with sets. Slower."
       OFF)

IF (${DISABLE_TR1})
  MESSAGE("-- Replacing unordered_map/set with map/set (warning: slower!)")
  ADD_DEFINITIONS(-DCERES_NO_TR1)
ELSE (${DISABLE_TR1})
  MESSAGE("-- Using normal TR1 unordered_map/set")
  # Use the std namespace for the hash<> and related templates. This may vary by
  # system.
  IF (MSVC)
    # This is known to work with Visual Studio 2010 Express.
    ADD_DEFINITIONS("\"-DCERES_HASH_NAMESPACE_START=namespace std {\"")
    ADD_DEFINITIONS("\"-DCERES_HASH_NAMESPACE_END=}\"")
  ELSE (MSVC)
    # This is known to work with recent versions of Linux and Mac OS X.
    ADD_DEFINITIONS("\"-DCERES_HASH_NAMESPACE_START=namespace std { namespace tr1 {\"")
    ADD_DEFINITIONS("\"-DCERES_HASH_NAMESPACE_END=}}\"")
  ENDIF (MSVC)
ENDIF (${DISABLE_TR1})

INCLUDE_DIRECTORIES(
  include
  internal
  internal/ceres
  ${GLOG_INCLUDE}
  ${EIGEN_INCLUDE}
  )

FILE(GLOB CERES_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/include/ceres/*.h)
#INSTALL(FILES ${CERES_HDRS} DESTINATION include/ceres)

FILE(GLOB CERES_PUBLIC_INTERNAL_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/include/ceres/internal/*.h)
#INSTALL(FILES ${CERES_PUBLIC_INTERNAL_HDRS} DESTINATION include/ceres/internal)

IF (${SUITESPARSE})
  INCLUDE_DIRECTORIES(${AMD_INCLUDE})
  INCLUDE_DIRECTORIES(${CAMD_INCLUDE})
  INCLUDE_DIRECTORIES(${COLAMD_INCLUDE})
  INCLUDE_DIRECTORIES(${CCOLAMD_INCLUDE})
  INCLUDE_DIRECTORIES(${CHOLMOD_INCLUDE})
  IF (${SUITESPARSE_CONFIG_FOUND})
    INCLUDE_DIRECTORIES(${SUITESPARSE_CONFIG_INCLUDE})
  ENDIF (${SUITESPARSE_CONFIG_FOUND})
  IF (${UFCONFIG_FOUND})
    INCLUDE_DIRECTORIES(${UFCONFIG_INCLUDE})
  ENDIF (${UFCONFIG_FOUND})
ENDIF(${SUITESPARSE})

IF (${CXSPARSE})
  INCLUDE_DIRECTORIES(${CXSPARSE_INCLUDE})
ENDIF(${CXSPARSE})

IF (${GFLAGS})
  INCLUDE_DIRECTORIES(${GFLAGS_INCLUDE})
ENDIF (${GFLAGS})

# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.
IF (NOT CMAKE_BUILD_TYPE)
  MESSAGE("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
ELSE (NOT CMAKE_BUILD_TYPE)
  IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE("\n=================================================================================")
    MESSAGE("\n-- Build type: Debug. Performance will be terrible!")
    MESSAGE("-- Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
    MESSAGE("\n=================================================================================")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  ENDIF (CMAKE_BUILD_TYPE STREQUAL "Debug")
ENDIF (NOT CMAKE_BUILD_TYPE)

# Set the default Ceres flags to an empty string.
SET (CERES_CXX_FLAGS)

IF (CMAKE_BUILD_TYPE STREQUAL "Release")
  IF (CMAKE_COMPILER_IS_GNUCXX)
    IF (${BUILD_ANDROID})
      # TODO(keir): Figure out what flags should go here to make an optimized
      # native ARM binary for Android.
    ELSE (${BUILD_ANDROID})
      # Linux
      IF (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        SET (CERES_CXX_FLAGS "${CERES_CXX_FLAGS} -march=native -mtune=native -fPIC")
      ENDIF (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
      # Mac OS X
      IF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        SET (CERES_CXX_FLAGS "${CERES_CXX_FLAGS} -fast -msse3")
      ENDIF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    ENDIF (${BUILD_ANDROID})
  ENDIF (CMAKE_COMPILER_IS_GNUCXX)
  IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    SET(CERES_CXX_FLAGS "${CERES_CXX_FLAGS} -O4")
  ENDIF()
ENDIF (CMAKE_BUILD_TYPE STREQUAL "Release")

SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CERES_CXX_FLAGS}")

# After the tweaks for the compile settings, disable some warnings on MSVC.
IF (MSVC)
  # Disable signed/unsigned int conversion warnings.
  ADD_DEFINITIONS("/wd4018")
  # Disable warning about using struct/class for the same symobl.
  ADD_DEFINITIONS("/wd4099")
  # Disable warning about the insecurity of using "std::copy".
  ADD_DEFINITIONS("/wd4996")
  # Disable performance warning about int-to-bool conversion.
  ADD_DEFINITIONS("/wd4800")
  # Disable performance warning about fopen insecurity.
  ADD_DEFINITIONS("/wd4996")
  # Disable warning about int64 to int32 conversion. Disabling
  # this warning may not be correct; needs investigation.
  # TODO(keir): Investigate these warnings in more detail.
  ADD_DEFINITIONS("/wd4244")
  # It's not possible to use STL types in DLL interfaces in a portable and
  # reliable way. However, that's what happens with Google Log and Google Flags
  # on Windows. MSVC gets upset about this and throws warnings that we can't do
  # much about. The real solution is to link static versions of Google Log and
  # Google Test, but that seems tricky on Windows. So, disable the warning.
  ADD_DEFINITIONS("/wd4251")

  # Google Flags doesn't have their DLL import/export stuff set up correctly,
  # which results in linker warnings. This is irrelevant for Ceres, so ignore
  # the warnings.
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4049")

  # Tuple sizes of 10 are used by Gtest.
  ADD_DEFINITIONS("-D_VARIADIC_MAX=10")
ENDIF (MSVC)

# GCC is not strict enough by default, so enable most of the warnings.
IF ("${UNIX}")
  SET(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-local-typedefs")
ENDIF ("${UNIX}")

# Use a larger inlining threshold for Clang, since it hobbles Eigen,
# resulting in an unreasonably slow version of the blas routines. The
# -Qunused-arguments is needed because CMake passes the inline
# threshold to the linker and clang complains about it and dies.
IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  SET(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Qunused-arguments -mllvm -inline-threshold=600 -Wno-return-type-c-linkage")
ENDIF()

ADD_SUBDIRECTORY(internal/ceres)

OPTION(BUILD_DOCUMENTATION
       "Build User's Guide (html)"
       OFF)

IF (${BUILD_DOCUMENTATION})
  MESSAGE("-- Documentation building is enabled")

  # Make CMake aware of the cmake folder, in order to find 'FindSphinx.cmake'
  SET (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

  # Generate the User's Guide (html).
  # The corresponding target is UserGuide, but is included in ALL.
  ADD_SUBDIRECTORY(docs)
ENDIF (${BUILD_DOCUMENTATION})

#OPTION(BUILD_EXAMPLES "Build examples" ON)
#
#IF (${BUILD_EXAMPLES})
#  MESSAGE("-- Build the examples.")
#  ADD_SUBDIRECTORY(examples)
#ELSE (${BUILD_EXAMPLES})
#  MESSAGE("-- Do not build any example.")
#ENDIF (${BUILD_EXAMPLES})

IF (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  # Add an uninstall target to remove all installed files.
  CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/cmake/uninstall.cmake"
                 IMMEDIATE @ONLY)

  ADD_CUSTOM_TARGET(uninstall
                    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake/uninstall.cmake)

  # Set up install directories. INCLUDE_INSTALL_DIR and LIB_INSTALL_DIR
  # must not be absolute paths.
  SET(LIB_INSTALL_DIR_SUFFIX "" CACHE
      STRING "The directories where to install libraries to")
  SET(LIB_INSTALL_DIR lib${LIB_INSTALL_DIR_SUFFIX} )
  SET(CMAKECONFIG_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/Ceres)
  SET(INCLUDE_INSTALL_DIR include)

  # This "exports" all targets which have been put into the export set
  # "CeresExport". This means that CMake generates a file with the given
  # filename, which can later on be loaded by projects using this package.
  # This file contains ADD_LIBRARY(bar IMPORTED) statements for each target
  # in the export set, so when loaded later on CMake will create "imported"
  # library targets from these, which can be used in many ways in the same way
  # as a normal library target created via a normal ADD_LIBRARY().
  INSTALL(EXPORT CeresExport
          DESTINATION ${CMAKECONFIG_INSTALL_DIR} FILE CeresTargets.cmake)

  # Figure out the relative path from the installed Config.cmake file to the
  # install prefix (which may be at runtime different from the chosen
  # CMAKE_INSTALL_PREFIX if under Windows the package was installed anywhere)
  # This relative path will be configured into the CeresConfig.cmake.
  FILE(RELATIVE_PATH relInstallDir
       ${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR} ${CMAKE_INSTALL_PREFIX})

  # Create a CeresConfig.cmake file. <name>Config.cmake files are searched by
  # FIND_PACKAGE() automatically. We configure that file so that we can put any
  # information we want in it, e.g. version numbers, include directories, etc.
  CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CeresConfig.cmake.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/CeresConfig.cmake" @ONLY)

  # Additionally, when CMake has found a CeresConfig.cmake, it can check for a
  # CeresConfigVersion.cmake in the same directory when figuring out the version
  # of the package when a version has been specified in the FIND_PACKAGE() call,
  # e.g. FIND_PACKAGE(Ceres [1.4.2] REQUIRED). The version argument is optional.
  CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CeresConfigVersion.cmake.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/CeresConfigVersion.cmake" @ONLY)

  # Install these two files into the same directory as the generated exports-file.
  INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/CeresConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/CeresConfigVersion.cmake"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/depend.cmake"
          DESTINATION ${CMAKECONFIG_INSTALL_DIR})
ENDIF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
