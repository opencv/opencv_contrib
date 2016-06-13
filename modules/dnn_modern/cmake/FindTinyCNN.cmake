# Locate the tiny-cnn library.
#
# Defines the following variables:
#
#   TinyCNN_FOUND        - TRUE if the tiny-cnn headers are found
#   TINYCNN_INCLUDE_DIRS - The path to tiny-cnn headers
#
# Accepts the following variables as input:
#
#   TinyCNN_ROOT - (as a CMake or environment variable)
#                  The root directory of the tiny-cnn install prefix

message(STATUS "Looking for tiny_cnn.h")

set(TINYCNN_INCLUDE_SEARCH_PATHS
    /usr/include/tiny_cnn
    /usr/local/include/tiny_cnn
    /opt/tiny_cnn
    $ENV{TINYCNN_ROOT}
    ${TINYCNN_ROOT}
    ${TINYCNN_ROOT}/tiny_cnn
)

find_path(TINYCNN_INCLUDE_DIR
    NAMES tiny_cnn/tiny_cnn.h
    HINTS ${TINYCNN_INCLUDE_SEARCH_PATHS}
)

# handle the QUIETLY and REQUIRED arguments and set TinyCNN_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TinyCNN
    FOUND_VAR TinyCNN_FOUND
    REQUIRED_VARS TINYCNN_INCLUDE_DIR)

if(TinyCNN_FOUND)
    set(TINYCNN_INCLUDE_DIRS ${TINYCNN_INCLUDE_DIR})
    message(STATUS "Looking for tiny_cnn.h - found")
    message(STATUS "Found tiny-cnn in: ${TINYCNN_INCLUDE_DIRS}")
else()
    message(STATUS "Looking for tiny_cnn.h - not found")
endif()

mark_as_advanced(
    TINYCNN_INCLUDE_DIRS
)