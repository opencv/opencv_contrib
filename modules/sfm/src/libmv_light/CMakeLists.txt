# installation rules
include(CMake/Installation.cmake)

set(BUILD_SHARED_LIBS OFF) # Force static libs for 3rdparty dependencies

ocv_warnings_disable(CMAKE_CXX_FLAGS -Winconsistent-missing-override -Wsuggest-override)

add_subdirectory(libmv)