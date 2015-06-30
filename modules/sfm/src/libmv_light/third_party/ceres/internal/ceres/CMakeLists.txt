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

SET(CERES_INTERNAL_SRC
    array_utils.cc
    block_evaluate_preparer.cc
    block_jacobi_preconditioner.cc
    block_jacobian_writer.cc
    block_random_access_dense_matrix.cc
    block_random_access_matrix.cc
    block_random_access_sparse_matrix.cc
    block_sparse_matrix.cc
    block_structure.cc
    canonical_views_clustering.cc
    cgnr_solver.cc
    compressed_row_jacobian_writer.cc
    compressed_row_sparse_matrix.cc
    conditioned_cost_function.cc
    conjugate_gradients_solver.cc
    coordinate_descent_minimizer.cc
    corrector.cc
    cxsparse.cc
    dense_normal_cholesky_solver.cc
    dense_qr_solver.cc
    dense_sparse_matrix.cc
    detect_structure.cc
    dogleg_strategy.cc
    evaluator.cc
    file.cc
    gradient_checking_cost_function.cc
    implicit_schur_complement.cc
    iterative_schur_complement_solver.cc
    levenberg_marquardt_strategy.cc
    line_search.cc
    line_search_direction.cc
    line_search_minimizer.cc
    linear_least_squares_problems.cc
    linear_operator.cc
    linear_solver.cc
    local_parameterization.cc
    loss_function.cc
    low_rank_inverse_hessian.cc
    minimizer.cc
    normal_prior.cc
    parameter_block_ordering.cc
    partitioned_matrix_view.cc
    polynomial.cc
    preconditioner.cc
    problem.cc
    problem_impl.cc
    program.cc
    residual_block.cc
    residual_block_utils.cc
    runtime_numeric_diff_cost_function.cc
    schur_complement_solver.cc
    schur_eliminator.cc
    schur_jacobi_preconditioner.cc
    scratch_evaluate_preparer.cc
    solver.cc
    solver_impl.cc
    sparse_matrix.cc
    sparse_normal_cholesky_solver.cc
    split.cc
    stringprintf.cc
    suitesparse.cc
    triplet_sparse_matrix.cc
    trust_region_minimizer.cc
    trust_region_strategy.cc
    types.cc
    visibility.cc
    visibility_based_preconditioner.cc
    wall_time.cc
)

# Heuristic for determining LIB_SUFFIX. FHS recommends that 64-bit systems
# install native libraries to lib64 rather than lib. Most distros seem to
# follow this convention with a couple notable exceptions (Debian-based and
# Arch-based distros) which we try to detect here.
IF (${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND
    NOT DEFINED LIB_SUFFIX AND
    NOT CMAKE_CROSSCOMPILING AND
    ${CMAKE_SIZEOF_VOID_P} EQUAL "8" AND
    NOT EXISTS "/etc/debian_version" AND
    NOT EXISTS "/etc/arch-release")
  SET(LIB_SUFFIX "64")
ENDIF()

If (${PROTOBUF_FOUND})
  PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS matrix.proto)
ENDIF (${PROTOBUF_FOUND})

# Also depend on the header files so that they appear in IDEs.
FILE(GLOB CERES_INTERNAL_HDRS *.h)

# Include the specialized schur solvers.
IF (${SCHUR_SPECIALIZATIONS})
  FILE(GLOB CERES_INTERNAL_SCHUR_FILES generated/*.cc)
ELSE (${SCHUR_SPECIALIZATIONS})
  # Only the fully dynamic solver. The build is much faster this way.
  FILE(GLOB CERES_INTERNAL_SCHUR_FILES generated/schur_eliminator_d_d_d.cc)
ENDIF (${SCHUR_SPECIALIZATIONS})

# For Android, use the internal Glog implementation.
IF (${BUILD_ANDROID})
  ADD_LIBRARY(miniglog STATIC
              miniglog/glog/logging.cc)

  # The Android logging library that defines e.g. __android_log_print is
  # creatively named "log".
  TARGET_LINK_LIBRARIES(miniglog log)

  INSTALL(TARGETS miniglog
          EXPORT  CeresExport
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib${LIB_SUFFIX}
          ARCHIVE DESTINATION lib${LIB_SUFFIX})
ENDIF (${BUILD_ANDROID})

SET(CERES_LIBRARY_DEPENDENCIES ${GLOG_LIB})

IF (${GFLAGS})
  LIST(APPEND CERES_LIBRARY_DEPENDENCIES ${GFLAGS_LIB})
ENDIF (${GFLAGS})

IF (${SUITESPARSE_FOUND})
  LIST(APPEND CERES_LIBRARY_DEPENDENCIES ${CHOLMOD_LIB})
  LIST(APPEND CERES_LIBRARY_DEPENDENCIES ${CCOLAMD_LIB})
  LIST(APPEND CERES_LIBRARY_DEPENDENCIES ${CAMD_LIB})
  LIST(APPEND CERES_LIBRARY_DEPENDENCIES ${COLAMD_LIB})
  LIST(APPEND CERES_LIBRARY_DEPENDENCIES ${AMD_LIB})
  IF (EXISTS ${SUITESPARSE_CONFIG_LIB})
    LIST(APPEND CERES_LIBRARY_DEPENDENCIES ${SUITESPARSE_CONFIG_LIB})
  ENDIF (EXISTS ${SUITESPARSE_CONFIG_LIB})

  IF (EXISTS ${METIS_LIB})
    LIST(APPEND CERES_LIBRARY_DEPENDENCIES ${METIS_LIB})
  ENDIF (EXISTS ${METIS_LIB})

  LIST(APPEND CERES_LIBRARY_DEPENDENCIES ${LAPACK_LIB})

  IF (EXISTS ${BLAS_LIB})
    LIST(APPEND CERES_LIBRARY_DEPENDENCIES ${BLAS_LIB})
  ENDIF (EXISTS ${BLAS_LIB})
ENDIF (${SUITESPARSE_FOUND})

IF (${CXSPARSE_FOUND})
  LIST(APPEND CERES_LIBRARY_DEPENDENCIES ${CXSPARSE_LIB})
ENDIF (${CXSPARSE_FOUND})

IF (${OPENMP_FOUND})
  IF (NOT MSVC)
    LIST(APPEND CERES_LIBRARY_DEPENDENCIES gomp)
  ENDIF (NOT MSVC)
ENDIF (${OPENMP_FOUND})

IF (${PROTOBUF_FOUND})
  LIST(APPEND CERES_LIBRARY_DEPENDENCIES ${PROTOBUF_LIBRARY})
ENDIF (${PROTOBUF_FOUND})

SET(CERES_LIBRARY_SOURCE
    ${PROTO_SRCS}
    ${PROTO_HDRS}
    ${CERES_INTERNAL_SRC}
    ${CERES_INTERNAL_HDRS}
    ${CERES_INTERNAL_SCHUR_FILES})

ADD_LIBRARY(ceres STATIC ${CERES_LIBRARY_SOURCE})
TARGET_LINK_LIBRARIES(ceres ${CERES_LIBRARY_DEPENDENCIES})

#INSTALL(TARGETS ceres
#        EXPORT  CeresExport
#        RUNTIME DESTINATION bin
#        LIBRARY DESTINATION lib${LIB_SUFFIX}
#        ARCHIVE DESTINATION lib${LIB_SUFFIX})

# Don't build a DLL on MSVC. Supporting Ceres as a DLL on Windows involves
# nontrivial changes that we haven't made yet.
IF (NOT MSVC AND NOT ${BUILD_ANDROID} AND ${BUILD_SHARED_LIBS})
  ADD_LIBRARY(ceres_shared SHARED ${CERES_LIBRARY_SOURCE})
  TARGET_LINK_LIBRARIES(ceres_shared ${CERES_LIBRARY_DEPENDENCIES})
  SET_TARGET_PROPERTIES(ceres_shared PROPERTIES
                        VERSION ${CERES_VERSION}
                        SOVERSION ${CERES_ABI_VERSION})

  #INSTALL(TARGETS ceres_shared
  #        EXPORT  CeresExport
  #        RUNTIME DESTINATION bin
  #        LIBRARY DESTINATION lib${LIB_SUFFIX}
  #        ARCHIVE DESTINATION lib${LIB_SUFFIX})

ENDIF (NOT MSVC AND NOT ${BUILD_ANDROID} AND ${BUILD_SHARED_LIBS})

IF (${BUILD_TESTING} AND ${GFLAGS})
  ADD_LIBRARY(gmock_gtest gmock_gtest_all.cc gmock_main.cc)
  ADD_LIBRARY(test_util
              evaluator_test_utils.cc
              numeric_diff_test_utils.cc
              test_util.cc)

  TARGET_LINK_LIBRARIES(gmock_gtest ${GFLAGS_LIB} ${GLOG_LIB})
  TARGET_LINK_LIBRARIES(test_util ${GFLAGS_LIB} ${GLOG_LIB} ceres gmock_gtest)

  MACRO (CERES_TEST NAME)
    get_filename_component(CERES_TESTS_DATA
                           "${CMAKE_CURRENT_SOURCE_DIR}/../../data"
                           ABSOLUTE)

    ADD_EXECUTABLE(ceres_${NAME}_test ${NAME}_test.cc)
    TARGET_LINK_LIBRARIES(ceres_${NAME}_test test_util ceres gmock_gtest)
    SET_TARGET_PROPERTIES(ceres_${NAME}_test PROPERTIES
       RUNTIME_OUTPUT_DIRECTORY         ${LIBMV_TESTS_OUTPUT_DIR}/ceres
       RUNTIME_OUTPUT_DIRECTORY_RELEASE ${LIBMV_TESTS_OUTPUT_DIR}/ceres
       RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${LIBMV_TESTS_OUTPUT_DIR}/ceres
       OUTPUT_NAME ${NAME}_test)
    ADD_TEST(NAME ${NAME}_test
             COMMAND ${LIBMV_TESTS_OUTPUT_DIR}/ceres/${NAME}_test
             --test_srcdir
             ${CERES_TESTS_DATA})
  ENDMACRO (CERES_TEST)

  CERES_TEST(array_utils)
  CERES_TEST(autodiff)
  CERES_TEST(autodiff_cost_function)
  CERES_TEST(autodiff_local_parameterization)
  CERES_TEST(blas)
  CERES_TEST(block_random_access_dense_matrix)
  CERES_TEST(block_random_access_sparse_matrix)
  CERES_TEST(block_sparse_matrix)
  CERES_TEST(canonical_views_clustering)
  CERES_TEST(compressed_row_sparse_matrix)
  CERES_TEST(conditioned_cost_function)
  CERES_TEST(corrector)
  CERES_TEST(cost_function_to_functor)
  CERES_TEST(dense_sparse_matrix)
  CERES_TEST(dynamic_autodiff_cost_function)
  CERES_TEST(evaluator)
  CERES_TEST(gradient_checker)
  CERES_TEST(gradient_checking_cost_function)
  CERES_TEST(graph)
  CERES_TEST(graph_algorithms)
  CERES_TEST(implicit_schur_complement)
  CERES_TEST(iterative_schur_complement_solver)
  CERES_TEST(jet)
  CERES_TEST(levenberg_marquardt_strategy)
  CERES_TEST(dogleg_strategy)
  CERES_TEST(local_parameterization)
  CERES_TEST(loss_function)
  CERES_TEST(minimizer)
  CERES_TEST(normal_prior)
  CERES_TEST(numeric_diff_cost_function)
  CERES_TEST(numeric_diff_functor)
  CERES_TEST(ordered_groups)
  CERES_TEST(parameter_block)
  CERES_TEST(parameter_block_ordering)
  CERES_TEST(partitioned_matrix_view)
  CERES_TEST(polynomial)
  CERES_TEST(problem)
  CERES_TEST(residual_block)
  CERES_TEST(residual_block_utils)
  CERES_TEST(rotation)
  CERES_TEST(runtime_numeric_diff_cost_function)
  CERES_TEST(schur_complement_solver)
  CERES_TEST(schur_eliminator)
  CERES_TEST(solver_impl)

  IF (${SUITESPARSE_FOUND})
    CERES_TEST(suitesparse)
  ENDIF (${SUITESPARSE_FOUND})

  CERES_TEST(symmetric_linear_solver)
  CERES_TEST(triplet_sparse_matrix)
  CERES_TEST(trust_region_minimizer)
  CERES_TEST(unsymmetric_linear_solver)
  CERES_TEST(visibility)
  CERES_TEST(visibility_based_preconditioner)

  # Put the large end to end test last.
  CERES_TEST(system)
ENDIF (${BUILD_TESTING} AND ${GFLAGS})

LIBMV_INSTALL_LIB(ceres)
LIBMV_INSTALL_THIRD_PARTY_LIB(ceres_shared)
