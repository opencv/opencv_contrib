# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2015 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: alexs.mac@gmail.com (Alex Stewart)
#

# FindCXSparse.cmake - Find CXSparse libraries & dependencies.
#
# This module defines the following variables which should be referenced
# by the caller to use the library.
#
# CXSPARSE_FOUND: TRUE iff CXSparse and all dependencies have been found.
# CXSPARSE_INCLUDE_DIRS: Include directories for CXSparse.
# CXSPARSE_LIBRARIES: Libraries for CXSparse and all dependencies.
#
# CXSPARSE_VERSION: Extracted from cs.h.
# CXSPARSE_MAIN_VERSION: Equal to 3 if CXSPARSE_VERSION = 3.1.2
# CXSPARSE_SUB_VERSION: Equal to 1 if CXSPARSE_VERSION = 3.1.2
# CXSPARSE_SUBSUB_VERSION: Equal to 2 if CXSPARSE_VERSION = 3.1.2
#
# The following variables control the behaviour of this module:
#
# CXSPARSE_INCLUDE_DIR_HINTS: List of additional directories in which to
#                             search for CXSparse includes,
#                             e.g: /timbuktu/include.
# CXSPARSE_LIBRARY_DIR_HINTS: List of additional directories in which to
#                             search for CXSparse libraries, e.g: /timbuktu/lib.
#
# The following variables are also defined by this module, but in line with
# CMake recommended FindPackage() module style should NOT be referenced directly
# by callers (use the plural variables detailed above instead).  These variables
# do however affect the behaviour of the module via FIND_[PATH/LIBRARY]() which
# are NOT re-called (i.e. search for library is not repeated) if these variables
# are set with valid values _in the CMake cache_. This means that if these
# variables are set directly in the cache, either by the user in the CMake GUI,
# or by the user passing -DVAR=VALUE directives to CMake when called (which
# explicitly defines a cache variable), then they will be used verbatim,
# bypassing the HINTS variables and other hard-coded search locations.
#
# CXSPARSE_INCLUDE_DIR: Include directory for CXSparse, not including the
#                       include directory of any dependencies.
# CXSPARSE_LIBRARY: CXSparse library, not including the libraries of any
#                   dependencies.

# Reset CALLERS_CMAKE_FIND_LIBRARY_PREFIXES to its value when
# FindCXSparse was invoked.
macro(CXSPARSE_RESET_FIND_LIBRARY_PREFIX)
    if (MSVC)
        set(CMAKE_FIND_LIBRARY_PREFIXES "${CALLERS_CMAKE_FIND_LIBRARY_PREFIXES}")
    endif (MSVC)
endmacro(CXSPARSE_RESET_FIND_LIBRARY_PREFIX)

# Called if we failed to find CXSparse or any of it's required dependencies,
# unsets all public (designed to be used externally) variables and reports
# error message at priority depending upon [REQUIRED/QUIET/<NONE>] argument.
macro(CXSPARSE_REPORT_NOT_FOUND REASON_MSG)
    unset(CXSPARSE_FOUND)
    unset(CXSPARSE_INCLUDE_DIRS)
    unset(CXSPARSE_LIBRARIES)
    # Make results of search visible in the CMake GUI if CXSparse has not
    # been found so that user does not have to toggle to advanced view.
    mark_as_advanced(CLEAR CXSPARSE_INCLUDE_DIR
            CXSPARSE_LIBRARY)

    cxsparse_reset_find_library_prefix()

    # Note <package>_FIND_[REQUIRED/QUIETLY] variables defined by FindPackage()
    # use the camelcase library name, not uppercase.
    if (CXSparse_FIND_QUIETLY)
        message(STATUS "Failed to find CXSparse - " ${REASON_MSG} ${ARGN})
    elseif (CXSparse_FIND_REQUIRED)
        message(FATAL_ERROR "Failed to find CXSparse - " ${REASON_MSG} ${ARGN})
    else()
        # Neither QUIETLY nor REQUIRED, use no priority which emits a message
        # but continues configuration and allows generation.
        message("-- Failed to find CXSparse - " ${REASON_MSG} ${ARGN})
    endif ()
    return()
endmacro(CXSPARSE_REPORT_NOT_FOUND)

# Protect against any alternative find_package scripts for this library having
# been called previously (in a client project) which set CXSPARSE_FOUND, but not
# the other variables we require / set here which could cause the search logic
# here to fail.
unset(CXSPARSE_FOUND)

# Handle possible presence of lib prefix for libraries on MSVC, see
# also CXSPARSE_RESET_FIND_LIBRARY_PREFIX().
if (MSVC)
    # Preserve the caller's original values for CMAKE_FIND_LIBRARY_PREFIXES
    # s/t we can set it back before returning.
    set(CALLERS_CMAKE_FIND_LIBRARY_PREFIXES "${CMAKE_FIND_LIBRARY_PREFIXES}")
    # The empty string in this list is important, it represents the case when
    # the libraries have no prefix (shared libraries / DLLs).
    set(CMAKE_FIND_LIBRARY_PREFIXES "lib" "" "${CMAKE_FIND_LIBRARY_PREFIXES}")
endif (MSVC)

# Search user-installed locations first, so that we prefer user installs
# to system installs where both exist.
#
# TODO: Add standard Windows search locations for CXSparse.
list(APPEND CXSPARSE_CHECK_INCLUDE_DIRS
        /usr/local/include
        /usr/local/homebrew/include # Mac OS X
        /opt/local/var/macports/software # Mac OS X.
        /opt/local/include
        /usr/include)
list(APPEND CXSPARSE_CHECK_LIBRARY_DIRS
        /usr/local/lib
        /usr/local/homebrew/lib # Mac OS X.
        /opt/local/lib
        /usr/lib)

# Search supplied hint directories first if supplied.
find_path(CXSPARSE_INCLUDE_DIR
        NAMES cs.h
        PATHS ${CXSPARSE_INCLUDE_DIR_HINTS}
        ${CXSPARSE_CHECK_INCLUDE_DIRS}
        PATH_SUFFIXES suitesparse)
if (NOT CXSPARSE_INCLUDE_DIR OR
        NOT EXISTS ${CXSPARSE_INCLUDE_DIR})
    cxsparse_report_not_found(
            "Could not find CXSparse include directory, set CXSPARSE_INCLUDE_DIR "
            "to directory containing cs.h")
endif (NOT CXSPARSE_INCLUDE_DIR OR
        NOT EXISTS ${CXSPARSE_INCLUDE_DIR})

find_library(CXSPARSE_LIBRARY NAMES cxsparse
        PATHS ${CXSPARSE_LIBRARY_DIR_HINTS}
        ${CXSPARSE_CHECK_LIBRARY_DIRS})
if (NOT CXSPARSE_LIBRARY OR
        NOT EXISTS ${CXSPARSE_LIBRARY})
    cxsparse_report_not_found(
            "Could not find CXSparse library, set CXSPARSE_LIBRARY "
            "to full path to libcxsparse.")
endif (NOT CXSPARSE_LIBRARY OR
        NOT EXISTS ${CXSPARSE_LIBRARY})

# Mark internally as found, then verify. CXSPARSE_REPORT_NOT_FOUND() unsets
# if called.
set(CXSPARSE_FOUND TRUE)

# Extract CXSparse version from cs.h
if (CXSPARSE_INCLUDE_DIR)
    set(CXSPARSE_VERSION_FILE ${CXSPARSE_INCLUDE_DIR}/cs.h)
    if (NOT EXISTS ${CXSPARSE_VERSION_FILE})
        cxsparse_report_not_found(
                "Could not find file: ${CXSPARSE_VERSION_FILE} "
                "containing version information in CXSparse install located at: "
                "${CXSPARSE_INCLUDE_DIR}.")
    else (NOT EXISTS ${CXSPARSE_VERSION_FILE})
        file(READ ${CXSPARSE_INCLUDE_DIR}/cs.h CXSPARSE_VERSION_FILE_CONTENTS)

        string(REGEX MATCH "#define CS_VER [0-9]+"
                CXSPARSE_MAIN_VERSION "${CXSPARSE_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "#define CS_VER ([0-9]+)" "\\1"
                CXSPARSE_MAIN_VERSION "${CXSPARSE_MAIN_VERSION}")

        string(REGEX MATCH "#define CS_SUBVER [0-9]+"
                CXSPARSE_SUB_VERSION "${CXSPARSE_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "#define CS_SUBVER ([0-9]+)" "\\1"
                CXSPARSE_SUB_VERSION "${CXSPARSE_SUB_VERSION}")

        string(REGEX MATCH "#define CS_SUBSUB [0-9]+"
                CXSPARSE_SUBSUB_VERSION "${CXSPARSE_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "#define CS_SUBSUB ([0-9]+)" "\\1"
                CXSPARSE_SUBSUB_VERSION "${CXSPARSE_SUBSUB_VERSION}")

        # This is on a single line s/t CMake does not interpret it as a list of
        # elements and insert ';' separators which would result in 3.;1.;2 nonsense.
        set(CXSPARSE_VERSION "${CXSPARSE_MAIN_VERSION}.${CXSPARSE_SUB_VERSION}.${CXSPARSE_SUBSUB_VERSION}")
    endif (NOT EXISTS ${CXSPARSE_VERSION_FILE})
endif (CXSPARSE_INCLUDE_DIR)

# Catch the case when the caller has set CXSPARSE_LIBRARY in the cache / GUI and
# thus FIND_LIBRARY was not called, but specified library is invalid, otherwise
# we would report CXSparse as found.
# TODO: This regex for CXSparse library is pretty primitive, we use lowercase
#       for comparison to handle Windows using CamelCase library names, could
#       this check be better?
string(TOLOWER "${CXSPARSE_LIBRARY}" LOWERCASE_CXSPARSE_LIBRARY)
if (CXSPARSE_LIBRARY AND
        EXISTS ${CXSPARSE_LIBRARY} AND
        NOT "${LOWERCASE_CXSPARSE_LIBRARY}" MATCHES ".*cxsparse[^/]*")
    cxsparse_report_not_found(
            "Caller defined CXSPARSE_LIBRARY: "
            "${CXSPARSE_LIBRARY} does not match CXSparse.")
endif (CXSPARSE_LIBRARY AND
        EXISTS ${CXSPARSE_LIBRARY} AND
        NOT "${LOWERCASE_CXSPARSE_LIBRARY}" MATCHES ".*cxsparse[^/]*")

# Set standard CMake FindPackage variables if found.
if (CXSPARSE_FOUND)
    set(CXSPARSE_INCLUDE_DIRS ${CXSPARSE_INCLUDE_DIR})
    set(CXSPARSE_LIBRARIES ${CXSPARSE_LIBRARY})
endif (CXSPARSE_FOUND)

cxsparse_reset_find_library_prefix()

# Handle REQUIRED / QUIET optional arguments and version.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CXSparse
        REQUIRED_VARS CXSPARSE_INCLUDE_DIRS CXSPARSE_LIBRARIES
        VERSION_VAR CXSPARSE_VERSION)

# Only mark internal variables as advanced if we found CXSparse, otherwise
# leave them visible in the standard GUI for the user to set manually.
if (CXSPARSE_FOUND)
    mark_as_advanced(FORCE CXSPARSE_INCLUDE_DIR
            CXSPARSE_LIBRARY)
endif (CXSPARSE_FOUND)
