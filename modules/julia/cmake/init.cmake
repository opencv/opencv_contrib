OCV_OPTION(WITH_JULIA         "Include Julia support (opencv_contrib)"     OFF   IF (NOT ANDROID AND NOT IOS AND NOT WINRT AND NOT WIN32))

ocv_assert(OPENCV_INITIAL_PASS)

if(WITH_JULIA OR DEFINED Julia_FOUND)
  ocv_cmake_hook_append(STATUS_DUMP_EXTRA "${CMAKE_CURRENT_LIST_DIR}/hooks/STATUS_DUMP_EXTRA.cmake")
endif()

# --- Julia ---
if(WITH_JULIA AND NOT DEFINED Julia_FOUND)
  include(${CMAKE_CURRENT_LIST_DIR}/FindJulia.cmake)
  if(NOT Julia_FOUND)
    message(WARNING "Julia was not found. Disabling Julia bindings...")
    ocv_module_disable(julia)
  endif()

  # publish vars for status() dumper
  set(Julia_FOUND "${Julia_FOUND}" PARENT_SCOPE)
  set(Julia_EXECUTABLE "${Julia_EXECUTABLE}" PARENT_SCOPE)
  set(HAVE_JULIA "YES" CACHE STRING ADVANCED)

endif()
