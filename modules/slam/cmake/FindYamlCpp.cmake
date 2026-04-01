# FindYamlCpp.cmake - Find yaml-cpp library.
#
# This module defines the following variables:
#
# YAMLCPP_FOUND: TRUE if yaml-cpp is found.
# YAMLCPP_INCLUDE_DIRS: Include directories for yaml-cpp.
# YAMLCPP_LIBRARIES: Libraries required to link yaml-cpp.
#
# The following variables control the behaviour of this module:
#
# YAMLCPP_INCLUDE_DIR_HINTS: List of additional directories in which to
#                            search for yaml-cpp includes.
# YAMLCPP_LIBRARY_DIR_HINTS: List of additional directories in which to
#                            search for yaml-cpp libraries.

# Find include directory
string(REPLACE ":" ";" YAMLCPP_INCLUDE_DIR_HINTS "$ENV{YAMLCPP_INCLUDE_DIR_HINTS}")
string(REPLACE ":" ";" CPATH "$ENV{CPATH}")
string(REPLACE ":" ";" C_INCLUDE_PATH "$ENV{C_INCLUDE_PATH}")
string(REPLACE ":" ";" CPLUS_INCLUDE_PATH "$ENV{CPLUS_INCLUDE_PATH}")
list(APPEND YAMLCPP_CHECK_INCLUDE_DIRS
        ${YAMLCPP_INCLUDE_DIR_HINTS}
        ${CPATH}
        ${C_INCLUDE_PATH}
        ${CPLUS_INCLUDE_PATH}
        /opt/local/include
        /usr/local/include
        /usr/local/opt/include
        /usr/include)
find_path(YAMLCPP_INCLUDE_DIRS NAMES yaml.h
        PATH_SUFFIXES yaml-cpp
        PATHS ${YAMLCPP_CHECK_INCLUDE_DIRS})

# Find library
string(REPLACE ":" ";" YAMLCPP_LIBRARY_DIR_HINTS "$ENV{YAMLCPP_LIBRARY_DIR_HINTS}")
string(REPLACE ":" ";" LIBRARY_PATH "$ENV{LIBRARY_PATH}")
string(REPLACE ":" ";" LD_LIBRARY_PATH "$ENV{LD_LIBRARY_PATH}")
list(APPEND YAMLCPP_CHECK_LIBRARY_DIRS
        ${YAMLCPP_LIBRARY_DIR_HINTS}
        ${LIBRARY_PATH}
        ${LD_LIBRARY_PATH}
        /opt/local/lib
        /usr/local/lib
        /usr/local/opt/lib
        /usr/lib)
find_library(YAMLCPP_LIBRARIES NAMES yaml-cpp
        PATHS ${YAMLCPP_CHECK_LIBRARY_DIRS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(YamlCpp DEFAULT_MSG
        YAMLCPP_INCLUDE_DIRS YAMLCPP_LIBRARIES)

if(YAMLCPP_INCLUDE_DIRS AND YAMLCPP_LIBRARIES)
    message(STATUS "Found yaml-cpp library: ${YAMLCPP_LIBRARIES}")
    message(STATUS "Found yaml-cpp header in: ${YAMLCPP_INCLUDE_DIRS}")
    set(YAMLCPP_FOUND YES)
    mark_as_advanced(FORCE YAMLCPP_INCLUDE_DIRS YAMLCPP_LIBRARIES)
else()
    message(FATAL_ERROR "Failed to find yaml-cpp")
    set(YAMLCPP_FOUND NO)
    mark_as_advanced(CLEAR YAMLCPP_INCLUDE_DIRS YAMLCPP_LIBRARIES)
endif()
