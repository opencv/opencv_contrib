# Locate the tiny-dnn library.
#
# Defines the following variables:
#
#   TinyDNN_FOUND        - TRUE if the tiny-dnn headers are found
#   TINYDNN_INCLUDE_DIRS - The path to tiny-dnn headers
#
# Accepts the following variables as input:
#
#   TinyDNN_ROOT - (as a CMake or environment variable)
#                  The root directory of the tiny-dnn install prefix

message(STATUS "Looking for tiny_dnn.h")

set(TINYDNN_INCLUDE_SEARCH_PATHS
    /usr/include/tiny_dnn
    /usr/local/include/tiny_dnn
    /opt/tiny_dnn
    $ENV{TINYDNN_ROOT}
    ${TINYDNN_ROOT}
    ${TINYDNN_ROOT}/tiny_dnn
    ${TINY_DNN_CPP_ROOT}
)

find_path(TINYDNN_INCLUDE_DIR
    NAMES tiny_dnn/tiny_dnn.h
    HINTS ${TINYDNN_INCLUDE_SEARCH_PATHS}
)

# handle the QUIETLY and REQUIRED arguments and set TinyDNN_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TinyDNN
    FOUND_VAR TinyDNN_FOUND
    REQUIRED_VARS TINYDNN_INCLUDE_DIR)

if(TinyDNN_FOUND)
    set(TINYDNN_INCLUDE_DIRS ${TINYDNN_INCLUDE_DIR})
    message(STATUS "Looking for tiny_dnn.h - found")
    message(STATUS "Found tiny-dnn in: ${TINYDNN_INCLUDE_DIRS}")
else()
    message(STATUS "Looking for tiny_dnn.h - not found")
endif()

mark_as_advanced(
    TINYDNN_INCLUDE_DIRS
)
