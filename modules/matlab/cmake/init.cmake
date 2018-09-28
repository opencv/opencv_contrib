OCV_OPTION(WITH_MATLAB         "Include Matlab support (opencv_contrib)"     OFF   IF (NOT ANDROID AND NOT IOS AND NOT WINRT))

ocv_assert(OPENCV_INITIAL_PASS)

if(WITH_MATLAB OR DEFINED MATLAB_FOUND)
  ocv_cmake_hook_append(STATUS_DUMP_EXTRA "${CMAKE_CURRENT_LIST_DIR}/hooks/STATUS_DUMP_EXTRA.cmake")
endif()

# --- Matlab/Octave ---
if(WITH_MATLAB AND NOT DEFINED MATLAB_FOUND)
  include(${CMAKE_CURRENT_LIST_DIR}/OpenCVFindMatlab.cmake)
  if(NOT MATLAB_FOUND)
    message(WARNING "Matlab or compiler (mex) was not found. Disabling Matlab bindings...")
    ocv_module_disable(matlab)
  endif()

  # publish vars for status() dumper
  set(MATLAB_FOUND "${MATLAB_FOUND}" PARENT_SCOPE)
  set(MATLAB_MEX_SCRIPT "${MATLAB_MEX_SCRIPT}" PARENT_SCOPE)
endif()

if(NOT OPENCV_DOCS_INCLUDE_MATLAB)
  list(APPEND DOXYGEN_BLACKLIST "matlab")
  set(DOXYGEN_BLACKLIST "${DOXYGEN_BLACKLIST}" PARENT_SCOPE)
endif()
