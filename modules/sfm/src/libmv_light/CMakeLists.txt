# installation rules
include(CMake/Installation.cmake)

set(BUILD_SHARED_LIBS OFF) # Force static libs for 3rdparty dependencies

ocv_warnings_disable(CMAKE_CXX_FLAGS -Winconsistent-missing-override -Wsuggest-override)
if(CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wclass-memaccess)
endif()

add_subdirectory(libmv)