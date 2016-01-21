# installation rules
include(CMake/Installation.cmake)

set(BUILD_SHARED_LIBS OFF) # Force static libs for 3rdparty dependencies
add_subdirectory(libmv)