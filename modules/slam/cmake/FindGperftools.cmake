# FindGperftools.cmake - Find google-perftools library.
#
# This module defines the following variables:
#
# GPERFTOOLS_FOUND: TRUE if google-perftools is found.
# GPERFTOOLS_INCLUDE_DIRS: Include directories for google-perftools.
# GPERFTOOLS_LIBRARIES: Libraries required to link google-perftools.
#
# The following variables control the behaviour of this module:
#
# GPERFTOOLS_INCLUDE_DIR_HINTS: List of additional directories in which to
#                               search for google-perftools includes.
# GPERFTOOLS_LIBRARY_DIR_HINTS: List of additional directories in which to
#                               search for google-perftools libraries.

# Find include directory
string(REPLACE ":" ";" GPERFTOOLS_INCLUDE_DIR_HINTS "$ENV{GPERFTOOLS_INCLUDE_DIR_HINTS}")
string(REPLACE ":" ";" CPATH "$ENV{CPATH}")
string(REPLACE ":" ";" C_INCLUDE_PATH "$ENV{C_INCLUDE_PATH}")
string(REPLACE ":" ";" CPLUS_INCLUDE_PATH "$ENV{CPLUS_INCLUDE_PATH}")
list(APPEND GPERFTOOLS_CHECK_INCLUDE_DIRS
        ${GPERFTOOLS_INCLUDE_DIR_HINTS}
        ${CPATH}
        ${C_INCLUDE_PATH}
        ${CPLUS_INCLUDE_PATH}
        /opt/local/include
        /usr/local/include
        /usr/local/opt/include
        /usr/include)
find_path(GPERFTOOLS_INCLUDE_DIRS NAMES gperftools/profiler.h
        PATHS ${GPERFTOOLS_CHECK_INCLUDE_DIRS})

# Find library
string(REPLACE ":" ";" GPERFTOOLS_LIBRARY_DIR_HINTS "$ENV{GPERFTOOLS_LIBRARY_DIR_HINTS}")
string(REPLACE ":" ";" LIBRARY_PATH "$ENV{LIBRARY_PATH}")
string(REPLACE ":" ";" LD_LIBRARY_PATH "$ENV{LD_LIBRARY_PATH}")
list(APPEND GPERFTOOLS_CHECK_LIBRARY_DIRS
        ${GPERFTOOLS_LIBRARY_DIR_HINTS}
        ${LIBRARY_PATH}
        ${LD_LIBRARY_PATH}
        /opt/local/lib
        /usr/local/lib
        /usr/local/opt/lib
        /usr/lib)
find_library(GPERFTOOLS_LIBRARIES NAMES profiler
        PATHS ${GPERFTOOLS_CHECK_LIBRARY_DIRS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gperftools DEFAULT_MSG
        GPERFTOOLS_INCLUDE_DIRS GPERFTOOLS_LIBRARIES)

if(GPERFTOOLS_INCLUDE_DIRS AND GPERFTOOLS_LIBRARIES)
    message(STATUS "Found google-perftools library: ${GPERFTOOLS_LIBRARIES}")
    message(STATUS "Found google-perftools header in: ${GPERFTOOLS_INCLUDE_DIRS}")
    set(GPERFTOOLS_FOUND YES)
    mark_as_advanced(FORCE GPERFTOOLS_INCLUDE_DIRS GPERFTOOLS_LIBRARIES)
else()
    message(FATAL_ERROR "Failed to find google-perftools")
    set(GPERFTOOLS_FOUND NO)
    mark_as_advanced(CLEAR GPERFTOOLS_INCLUDE_DIRS GPERFTOOLS_LIBRARIES)
endif()
