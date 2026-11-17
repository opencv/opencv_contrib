# Optional, default-off GPU operations with runtime-compiled Triton kernels.
# Compilation uses a separate Python/Triton process; execution is C++ only.
# This module does not require PyTorch or compile kernels during the OpenCV build.
#
# TRITON_BACKEND selects a Linux backend:
#   cuda: CUDA driver API, GpuMat and DeviceImage entries, kernels/*_cpu.py.
#   hip:  ROCm/DTK driver API, DeviceImage entries, generated kernels/*_cpu_hip.py.
#         GpuMat entries report StsNotImplemented on this backend.
#   auto: CUDA when enabled in OpenCV, otherwise HIP when its toolkit is available.
# Select hip explicitly when CUDA is provided by a compatibility layer.
set(BUILD_opencv_triton_INIT OFF)
set(the_description "Triton-backed operations (runtime JIT)")
# Register first: OpenCV returns during discovery and skips disabled modules,
# so ordinary builds never probe Python/Triton or require a HIP/CUDA toolkit.
ocv_add_module(triton opencv_core opencv_imgproc WRAP python)

set(TRITON_BACKEND "auto" CACHE STRING "Device backend of the triton module: auto, cuda or hip")
set_property(CACHE TRITON_BACKEND PROPERTY STRINGS auto cuda hip)

# HIP toolkit. TRITON_ROCM_PATH is the toolkit root and doubles as the default ROCM_PATH the
# compile helper's compiler needs at run time; the environment's ROCM_PATH still wins then.
set(TRITON_ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm/DTK root, the compile helper's default ROCM_PATH (hip backend)")
find_path(TRITON_HIP_INCLUDE_DIR hip/hip_runtime_api.h
          HINTS "${TRITON_ROCM_PATH}" ENV ROCM_PATH ENV HIP_PATH
          PATHS /opt/dtk /opt/rocm
          PATH_SUFFIXES include hip/include)
find_library(TRITON_HIP_LIBRARY amdhip64
             HINTS "${TRITON_ROCM_PATH}" ENV ROCM_PATH ENV HIP_PATH
             PATHS /opt/dtk /opt/rocm
             PATH_SUFFIXES lib lib64 hip/lib)
mark_as_advanced(TRITON_HIP_INCLUDE_DIR TRITON_HIP_LIBRARY)
set(_triton_have_hip OFF)
if(TRITON_HIP_INCLUDE_DIR AND TRITON_HIP_LIBRARY)
  set(_triton_have_hip ON)
endif()

set(_triton_backend "none")
if(TRITON_BACKEND STREQUAL "auto")
  if(HAVE_CUDA)
    set(_triton_backend "cuda")
  elseif(_triton_have_hip)
    set(_triton_backend "hip")
  endif()
elseif(TRITON_BACKEND STREQUAL "cuda")
  if(HAVE_CUDA)
    set(_triton_backend "cuda")
  else()
    message(WARNING "triton: TRITON_BACKEND=cuda but this OpenCV build has no CUDA; the module is disabled")
  endif()
elseif(TRITON_BACKEND STREQUAL "hip")
  if(_triton_have_hip)
    set(_triton_backend "hip")
  else()
    message(WARNING "triton: TRITON_BACKEND=hip but hip/hip_runtime_api.h or libamdhip64 was not found "
                    "(set TRITON_ROCM_PATH or ROCM_PATH); the module is disabled")
  endif()
else()
  message(FATAL_ERROR "triton: TRITON_BACKEND must be auto, cuda or hip (got '${TRITON_BACKEND}')")
endif()

if(IOS OR WINRT OR WIN32 OR APPLE OR _triton_backend STREQUAL "none")
  ocv_module_disable(triton)
endif()
message(STATUS "triton: backend ${_triton_backend}")

option(TRITON_PERF_WITH_NATIVE_BASELINE "Build native CUDA comparisons into opencv_perf_triton" OFF)
if(TRITON_PERF_WITH_NATIVE_BASELINE)
  if(NOT BUILD_PERF_TESTS)
    message(FATAL_ERROR "triton: TRITON_PERF_WITH_NATIVE_BASELINE requires BUILD_PERF_TESTS=ON")
  endif()
  if(NOT _triton_backend STREQUAL "cuda")
    message(FATAL_ERROR "triton: TRITON_PERF_WITH_NATIVE_BASELINE currently requires the CUDA backend")
  endif()
endif()

# The interpreter the compile helper runs under, and the PYTHONPATH that makes `import triton`
# work for it. Distinct from PYTHON3_EXECUTABLE (the Python bindings' interpreter), though it
# defaults to it. Both can be overridden at run time (OPENCV_TRITON_PYTHON[PATH]).
set(TRITON_PYTHON_EXECUTABLE "${PYTHON3_EXECUTABLE}" CACHE FILEPATH "Python used by the Triton compile helper")
set(TRITON_PYTHONPATH "" CACHE STRING "PYTHONPATH for the Triton compile helper (where `triton` is importable)")

set(_triton_compiler "unknown")
if(TRITON_PYTHON_EXECUTABLE)
  execute_process(COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${TRITON_PYTHONPATH}"
                          "${TRITON_PYTHON_EXECUTABLE}" -c "import sys; sys.modules['torch'] = None; import triton; print(triton.__version__)"
                  OUTPUT_VARIABLE _triton_version OUTPUT_STRIP_TRAILING_WHITESPACE
                  RESULT_VARIABLE _triton_rc ERROR_QUIET)
  if(_triton_rc EQUAL 0 AND _triton_version)
    set(_triton_compiler "triton-${_triton_version}")
  endif()
endif()
if(_triton_compiler STREQUAL "unknown")
  message(WARNING "triton: `import triton` failed under ${TRITON_PYTHON_EXECUTABLE} (PYTHONPATH=${TRITON_PYTHONPATH}); "
                  "the module builds, but every cache miss will report the toolchain as unavailable")
endif()
message(STATUS "triton: compile helper ${TRITON_PYTHON_EXECUTABLE}, ${_triton_compiler}")

set(_triton_srcs
    "${CMAKE_CURRENT_LIST_DIR}/src/core_resize.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/entry.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/device_entries.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/python_entries.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/resize_plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/provider.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/cvtcolor.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/warpaffine.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/compile_protocol.cpp")
if(_triton_backend STREQUAL "hip")
  # The GpuMat entries need cv::cuda, which a hip build of OpenCV does not have: this file
  # defines the same symbols and raises, pointing at opencv2/triton/device.hpp.
  list(APPEND _triton_srcs
       "${CMAKE_CURRENT_LIST_DIR}/src/gpumat_adapter_unavailable.cpp"
       "${CMAKE_CURRENT_LIST_DIR}/src/runtime/hip_runtime.cpp")
  set(_triton_kernel_files resize_cpu_hip.py cvtcolor_cpu_hip.py warpaffine_cpu_hip.py)
else()
  list(APPEND _triton_srcs
       "${CMAKE_CURRENT_LIST_DIR}/src/cuda_resize_adapter.cpp"
       "${CMAKE_CURRENT_LIST_DIR}/src/runtime/cuda_runtime.cpp")
  set(_triton_kernel_files resize_cpu.py cvtcolor_cpu.py warpaffine_cpu.py)
endif()
# Install the public umbrella and caller-owned device-buffer headers below.
# src/core.hpp is private implementation support.
file(GLOB _triton_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/triton.hpp"
                       "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/triton/device.hpp")
file(GLOB _triton_int_hdrs "${CMAKE_CURRENT_LIST_DIR}/src/*.hpp" "${CMAKE_CURRENT_LIST_DIR}/src/runtime/*.hpp")

# imgproc: the interpolation enums. cudawarping is not a dependency of the implementation; it is
# the performance baseline of perf/ only (a different computation: cv::cuda semantics).
ocv_set_module_sources(HEADERS ${_triton_hdrs} SOURCES ${_triton_srcs} ${_triton_int_hdrs})
ocv_module_include_directories()
ocv_create_module()

if(_triton_backend STREQUAL "cuda")
  # --- CUDA driver API --------------------------------------------------------------------
  set(_triton_cuda_inc "")
  foreach(_v CUDAToolkit_INCLUDE_DIRS CUDA_INCLUDE_DIRS CUDA_TOOLKIT_INCLUDE)
    if(${_v})
      list(APPEND _triton_cuda_inc ${${_v}})
    endif()
  endforeach()
  if(NOT _triton_cuda_inc AND CUDAToolkit_ROOT)
    list(APPEND _triton_cuda_inc "${CUDAToolkit_ROOT}/include")
  endif()
  if(NOT _triton_cuda_inc)
    message(FATAL_ERROR "triton: could not resolve the CUDA include directory; pass -DCUDAToolkit_INCLUDE_DIRS=<path>")
  endif()
  set(_triton_cuda_lib "")
  if(CUDA_CUDA_LIBRARY)
    set(_triton_cuda_lib ${CUDA_CUDA_LIBRARY})
  elseif(TARGET CUDA::cuda_driver)
    set(_triton_cuda_lib CUDA::cuda_driver)
  else()
    set(_triton_cuda_lib cuda)
  endif()
  ocv_target_include_directories(${the_module} ${_triton_cuda_inc})
  ocv_target_link_libraries(${the_module} LINK_PRIVATE ${_triton_cuda_lib} ${CMAKE_DL_LIBS})
  ocv_target_compile_definitions(${the_module} PRIVATE OPENCV_TRITON_HAVE_CUDA_ADAPTER=1)
else()
  # --- HIP (DTK / ROCm) -----------------------------------------------------------------
  # SYSTEM: the toolkit headers come after OpenCV's own, and their deprecation noise stays out.
  # __HIP_PLATFORM_AMD__ is what hipcc defines implicitly and a plain C++ compiler must be told.
  set(_triton_rocm "${TRITON_ROCM_PATH}")
  if(NOT _triton_rocm)
    get_filename_component(_triton_rocm "${TRITON_HIP_INCLUDE_DIR}" DIRECTORY)
  endif()
  ocv_target_include_directories(${the_module} SYSTEM "${TRITON_HIP_INCLUDE_DIR}")
  ocv_target_link_libraries(${the_module} LINK_PRIVATE "${TRITON_HIP_LIBRARY}" ${CMAKE_DL_LIBS})
  ocv_target_compile_definitions(${the_module} PRIVATE
    __HIP_PLATFORM_AMD__=1 OPENCV_TRITON_HAVE_HIP_ADAPTER=1
    OPENCV_TRITON_ROCM_PATH_DEFAULT="${_triton_rocm}")
  message(STATUS "triton: hip ${TRITON_HIP_INCLUDE_DIR}, ${TRITON_HIP_LIBRARY}, ROCM_PATH default ${_triton_rocm}")

  # The hip kernels are derived from the CUDA sources; a stale copy would serve kernels that
  # differ from what the CUDA source defines, so drift fails the configure.
  if(TRITON_PYTHON_EXECUTABLE)
    execute_process(COMMAND "${TRITON_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tools/make_kernels_cpu_hip.py" --check
                    RESULT_VARIABLE _triton_hip_rc OUTPUT_QUIET ERROR_VARIABLE _triton_hip_err)
    if(NOT _triton_hip_rc EQUAL 0)
      message(FATAL_ERROR "triton: ${_triton_hip_err}")
    endif()
    # Same rule for the runtime: src/runtime/hip_runtime.cpp is generated from cuda_runtime.cpp.
    execute_process(COMMAND "${TRITON_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tools/port_hip_runtime.py" --check
                    RESULT_VARIABLE _triton_hip_rc OUTPUT_QUIET ERROR_VARIABLE _triton_hip_err)
    if(NOT _triton_hip_rc EQUAL 0)
      message(FATAL_ERROR "triton: ${_triton_hip_err}")
    endif()
  else()
    message(WARNING "triton: no Python to check that kernels/*_cpu_hip.py and src/runtime/hip_runtime.cpp "
                    "are current (run tools/make_kernels_cpu_hip.py --check, tools/port_hip_runtime.py --check)")
  endif()
endif()

# The plan tables are float32 arithmetic that must round op by op, as the reference does.
# Source properties also apply when these files are collected into opencv_world.
set_property(SOURCE ${_triton_srcs} APPEND PROPERTY COMPILE_OPTIONS -ffp-contract=off)

if(NOT BUILD_SHARED_LIBS)
  ocv_target_compile_definitions(${the_module} PRIVATE OPENCV_TRITON_STATIC_BUILD=1)
endif()

# --- resources: kernels source + compile helper -----------------------------------------
# Installed under <share>/triton and found at run time relative to the library, so an install
# tree can be moved; an uninstalled build falls back to this source directory.
set(_triton_res_install "${OPENCV_OTHER_INSTALL_PATH}/triton")
file(RELATIVE_PATH _triton_rel "${CMAKE_INSTALL_PREFIX}/${OPENCV_LIB_INSTALL_PATH}" "${CMAKE_INSTALL_PREFIX}/${_triton_res_install}")
ocv_target_compile_definitions(${the_module} PRIVATE
  OPENCV_TRITON_REL_RESOURCES="${_triton_rel}"
  OPENCV_TRITON_INSTALL_RESOURCES="${CMAKE_INSTALL_PREFIX}/${_triton_res_install}"
  OPENCV_TRITON_SOURCE_RESOURCES="${CMAKE_CURRENT_LIST_DIR}"
  OPENCV_TRITON_PYTHON_DEFAULT="${TRITON_PYTHON_EXECUTABLE}"
  OPENCV_TRITON_PYTHONPATH_DEFAULT="${TRITON_PYTHONPATH}"
  OPENCV_TRITON_COMPILER_DEFAULT="${_triton_compiler}")
set(_triton_kernel_paths "")
foreach(_k ${_triton_kernel_files})
  list(APPEND _triton_kernel_paths "${CMAKE_CURRENT_LIST_DIR}/kernels/${_k}")
endforeach()
install(FILES ${_triton_kernel_paths} DESTINATION "${_triton_res_install}/kernels" COMPONENT libs)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/tools/compile_helper.py" DESTINATION "${_triton_res_install}/tools" COMPONENT libs)

# --- tests / perf / samples -------------------------------------------------------------
# Compile private validation support as separate translation units, without
# including implementation .cpp files or exporting private production symbols.
set(_triton_test_support_sources
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/provider.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/compile_protocol.cpp")
set(_triton_reference_sources
    "${CMAKE_CURRENT_LIST_DIR}/src/runtime/resize_plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/test/resize_reference.cpp")
if(_triton_backend STREQUAL "cuda")
  # OpenCV recursively globs test/ by default; HIP's standalone main/toolkit
  # sources must not be compiled into the CUDA gtest executable.
  file(GLOB _triton_cuda_test_sources "${CMAKE_CURRENT_LIST_DIR}/test/*.cpp"
                                     "${CMAKE_CURRENT_LIST_DIR}/test/*.hpp")
  # The evaluator and table builder are compiled explicitly below.
  list(REMOVE_ITEM _triton_cuda_test_sources "${CMAKE_CURRENT_LIST_DIR}/test/resize_reference.cpp")
  ocv_add_accuracy_tests(FILES "Src" ${_triton_cuda_test_sources}
      ${_triton_test_support_sources}
      "${CMAKE_CURRENT_LIST_DIR}/src/runtime/cvtcolor.cpp"
      "${CMAKE_CURRENT_LIST_DIR}/src/runtime/warpaffine.cpp")
  # Select top-level perf sources only; future backend support is selected here.
  file(GLOB _triton_common_perf_sources "${CMAKE_CURRENT_LIST_DIR}/perf/*.cpp"
                                       "${CMAKE_CURRENT_LIST_DIR}/perf/*.hpp")
  ocv_add_perf_tests(FILES "Src" ${_triton_common_perf_sources})
  ocv_add_samples()
  # Native comparisons are an explicit perf-only dependency, never auto-discovered.
  if(TRITON_PERF_WITH_NATIVE_BASELINE)
    if(NOT TARGET opencv_perf_triton)
      message(FATAL_ERROR "triton: native baselines requested, but opencv_perf_triton is not selected; "
                          "check BUILD_PERF_TESTS and OPENCV_BUILD_PERF_TEST_MODULES_LIST")
    endif()
    ocv_check_dependencies(opencv_cudawarping opencv_cudaimgproc)
    if(NOT OCV_DEPENDENCIES_FOUND)
      message(FATAL_ERROR "triton: native baselines require opencv_cudawarping and opencv_cudaimgproc; "
                          "include cudawarping,cudaimgproc in BUILD_LIST")
    endif()
    target_sources(opencv_perf_triton PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/perf/baseline/native_cuda_resize.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/perf/baseline/native_cuda_cvtcolor.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/perf/baseline/native_cuda_warpaffine.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/perf/baseline/native_cuda_precomp.hpp")
    ocv_target_include_modules(opencv_perf_triton opencv_cudawarping opencv_cudaimgproc)
    ocv_target_link_libraries(opencv_perf_triton LINK_PRIVATE opencv_cudawarping opencv_cudaimgproc)
  endif()
  foreach(_t opencv_test_triton opencv_perf_triton)
    if(TARGET ${_t})
      target_sources(${_t} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/validation/device_cuda.cpp" ${_triton_reference_sources})
      # Compile the CPU reference independently for the test and perf executables.
      target_compile_options(${_t} PRIVATE -ffp-contract=off)
      ocv_target_include_directories(${_t} ${_triton_cuda_inc})
      ocv_target_link_libraries(${_t} LINK_PRIVATE ${_triton_cuda_lib})
    endif()
  endforeach()
else()
  # Standard common accuracy/perf sources use the HIP implementation of the
  # private validation boundary. Driver-specific checks use separate targets.
  ocv_add_accuracy_tests(FILES "Src"
      "${CMAKE_CURRENT_LIST_DIR}/test/test_main.cpp"
      "${CMAKE_CURRENT_LIST_DIR}/test/test_common.cpp"
      "${CMAKE_CURRENT_LIST_DIR}/test/test_resize_reference.cpp"
      ${_triton_test_support_sources})
  file(GLOB _triton_common_perf_sources "${CMAKE_CURRENT_LIST_DIR}/perf/*.cpp"
                                      "${CMAKE_CURRENT_LIST_DIR}/perf/*.hpp")
  ocv_add_perf_tests(FILES "Src" ${_triton_common_perf_sources})
  foreach(_target opencv_test_triton opencv_perf_triton)
    if(TARGET ${_target})
      target_sources(${_target} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/validation/device_hip.cpp" ${_triton_reference_sources})
      # Perf also compares against the test-only CPU reference.
      if(_target STREQUAL "opencv_perf_triton")
        target_sources(${_target} PRIVATE ${_triton_test_support_sources})
      endif()
      target_include_directories(${_target} SYSTEM PRIVATE "${TRITON_HIP_INCLUDE_DIR}")
      target_compile_definitions(${_target} PRIVATE __HIP_PLATFORM_AMD__=1)
      target_compile_options(${_target} PRIVATE -ffp-contract=off)
      target_link_libraries(${_target} PRIVATE "${TRITON_HIP_LIBRARY}")
    endif()
  endforeach()
  # The device-buffer sample has no GUI or video dependencies.
  set(_triton_sample "${CMAKE_CURRENT_LIST_DIR}/samples/hip/device.cpp")
  if(BUILD_EXAMPLES)
    set(_sample_target example_triton_device)
    ocv_add_executable(${_sample_target} "${_triton_sample}")
    ocv_target_include_modules(${_sample_target} opencv_core opencv_imgproc ${the_module})
    ocv_target_include_directories(${_sample_target} SYSTEM "${TRITON_HIP_INCLUDE_DIR}")
    target_compile_definitions(${_sample_target} PRIVATE __HIP_PLATFORM_AMD__=1)
    ocv_target_link_libraries(${_sample_target} PRIVATE opencv_core opencv_imgproc ${the_module} "${TRITON_HIP_LIBRARY}")
    set_target_properties(${_sample_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
      PROJECT_LABEL "(sample) device"
      LABELS "${OPENCV_MODULE_${the_module}_LABEL};Sample")
    set_source_files_properties("${_triton_sample}" PROPERTIES
      LABELS "${OPENCV_MODULE_${the_module}_LABEL};Sample")
    if(ENABLE_SOLUTION_FOLDERS)
      set_target_properties(${_sample_target} PROPERTIES FOLDER "samples/triton")
    endif()
    if(NOT TARGET opencv_samples_triton)
      add_custom_target(opencv_samples_triton)
      add_dependencies(opencv_samples opencv_samples_triton)
    endif()
    add_dependencies(opencv_samples_triton ${_sample_target})
    if(INSTALL_BIN_EXAMPLES)
      install(TARGETS ${_sample_target} RUNTIME
        DESTINATION "${OPENCV_SAMPLES_BIN_INSTALL_PATH}/triton" COMPONENT samples)
    endif()
  endif()
  # Source installation is independent of whether example binaries are built.
  if(INSTALL_C_EXAMPLES)
    install(FILES "${_triton_sample}"
      DESTINATION "${OPENCV_SAMPLES_SRC_INSTALL_PATH}/triton/hip" COMPONENT samples)
  endif()

  option(TRITON_BUILD_HIP_TEST "Build backend-specific HIP lifecycle tests" ${BUILD_TESTS})
  if(BUILD_TESTS AND TRITON_BUILD_HIP_TEST
      AND (NOT DEFINED OPENCV_BUILD_TEST_MODULES_LIST
          OR OPENCV_BUILD_TEST_MODULES_LIST STREQUAL "all"
          OR ";${OPENCV_BUILD_TEST_MODULES_LIST};" MATCHES ";triton;"))
    set(_t opencv_test_triton_hip)
    set(_triton_hip_test_sources
      "${CMAKE_CURRENT_LIST_DIR}/test/hip/test_hip.cpp"
      ${_triton_test_support_sources}
      ${_triton_reference_sources})
    ocv_add_executable(${_t} ${_triton_hip_test_sources})
    ocv_target_include_modules(${_t} opencv_core opencv_imgproc ${the_module})
    ocv_target_include_directories(${_t} SYSTEM "${TRITON_HIP_INCLUDE_DIR}")
    target_compile_definitions(${_t} PRIVATE __HIP_PLATFORM_AMD__=1 __OPENCV_TESTS=1)
    # The oracle rounds each operation independently.
    target_compile_options(${_t} PRIVATE -ffp-contract=off)
    ocv_target_link_libraries(${_t} PRIVATE opencv_core opencv_imgproc ${the_module} "${TRITON_HIP_LIBRARY}" ${CMAKE_DL_LIBS})
    set_target_properties(${_t} PROPERTIES
      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
      RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
      LABELS "${OPENCV_MODULE_${the_module}_LABEL};AccuracyTest")
    set_source_files_properties(${_triton_hip_test_sources} PROPERTIES
      LABELS "${OPENCV_MODULE_${the_module}_LABEL};AccuracyTest")
    if(ENABLE_SOLUTION_FOLDERS)
      set_target_properties(${_t} PROPERTIES FOLDER "tests accuracy")
    endif()
    add_dependencies(opencv_tests ${_t})
    ocv_add_test_from_target("${_t}" "Accuracy" "${_t}")
    if(INSTALL_TESTS)
      install(TARGETS ${_t} RUNTIME DESTINATION "${OPENCV_TEST_INSTALL_PATH}" COMPONENT tests)
    endif()
  endif()
  # Internal graph replay timing, separate from ordinary-call performance.
  # The helper has no Torch dependency.
  option(TRITON_BUILD_HIP_GRAPH_BENCH "Build internal warmed production HIP graph benchmark" OFF)
  if(TRITON_BUILD_HIP_GRAPH_BENCH AND BUILD_PERF_TESTS
      AND (NOT DEFINED OPENCV_BUILD_PERF_TEST_MODULES_LIST
          OR OPENCV_BUILD_PERF_TEST_MODULES_LIST STREQUAL "all"
          OR ";${OPENCV_BUILD_PERF_TEST_MODULES_LIST};" MATCHES ";triton;"))
    set(_graph_target opencv_perf_triton_hip_graph)
    ocv_add_executable(${_graph_target}
      "${CMAKE_CURRENT_LIST_DIR}/perf/hip/perf_graph.cpp"
      "${CMAKE_CURRENT_LIST_DIR}/src/runtime/provider.cpp"
      "${CMAKE_CURRENT_LIST_DIR}/src/runtime/compile_protocol.cpp"
      ${_triton_reference_sources})
    ocv_target_include_modules(${_graph_target} opencv_core opencv_imgproc ${the_module})
    ocv_target_include_directories(${_graph_target} SYSTEM "${TRITON_HIP_INCLUDE_DIR}")
    target_compile_definitions(${_graph_target} PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(${_graph_target} PRIVATE -ffp-contract=off)
    ocv_target_link_libraries(${_graph_target} PRIVATE opencv_core opencv_imgproc ${the_module} "${TRITON_HIP_LIBRARY}" ${CMAKE_DL_LIBS})
    set_target_properties(${_graph_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
  endif()

endif()
