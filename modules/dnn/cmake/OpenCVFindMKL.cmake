#
# The script to detect Intel(R) Math Kernel Library (MKL)
# installation/package
#
# Parameters:
# MKL_WITH_TBB
#
# On return this will define:
#
# HAVE_MKL          - True if Intel IPP found
# MKL_ROOT_DIR      - root of IPP installation
# MKL_INCLUDE_DIRS  - IPP include folder
# MKL_LIBRARIES     - IPP libraries that are used by OpenCV
#

macro(mkl_fail)
    set(HAVE_MKL OFF CACHE BOOL "True if MKL found")
    set(MKL_ROOT_DIR ${MKL_ROOT_DIR} CACHE PATH "Path to MKL directory")
    unset(MKL_INCLUDE_DIRS CACHE)
    unset(MKL_LIBRARIES CACHE)
endmacro()

macro(get_mkl_version VERSION_FILE)
    # read MKL version info from file
    file(STRINGS ${VERSION_FILE} STR1 REGEX "__INTEL_MKL__")
    file(STRINGS ${VERSION_FILE} STR2 REGEX "__INTEL_MKL_MINOR__")
    file(STRINGS ${VERSION_FILE} STR3 REGEX "__INTEL_MKL_UPDATE__")
    #file(STRINGS ${VERSION_FILE} STR4 REGEX "INTEL_MKL_VERSION")

    # extract info and assign to variables
    string(REGEX MATCHALL "[0-9]+" MKL_VERSION_MAJOR ${STR1})
    string(REGEX MATCHALL "[0-9]+" MKL_VERSION_MINOR ${STR2})
    string(REGEX MATCHALL "[0-9]+" MKL_VERSION_UPDATE ${STR3})
    set(MKL_VERSION_STR "${MKL_VERSION_MAJOR}.${MKL_VERSION_MINOR}.${MKL_VERSION_UPDATE}" CACHE STRING "MKL version" FORCE)
endmacro()


if(NOT DEFINED MKL_USE_MULTITHREAD)
    OCV_OPTION(MKL_WITH_TBB "Use MKL with TBB multithreading" OFF)#ON IF WITH_TBB)
    OCV_OPTION(MKL_WITH_OPENMP "Use MKL with OpenMP multithreading" OFF)#ON IF WITH_OPENMP)
endif()

#check current MKL_ROOT_DIR
if(NOT MKL_ROOT_DIR OR NOT EXISTS ${MKL_ROOT_DIR}/include/mkl.h)
    set(MKLROOT_PATHS ${MKL_ROOT_DIR})
    if(DEFINED $ENV{MKLROOT})
        list(APPEND MKLROOT_PATHS $ENV{MKLROOT})
    endif()
    if(WIN32)
        set(ProgramFilesx86 "ProgramFiles(x86)")
        list(APPEND MKLROOT_PATHS $ENV{${ProgramFilesx86}}/IntelSWTools/compilers_and_libraries/windows/mkl)
    endif()
    if(UNIX)
        list(APPEND MKLROOT_PATHS "/opt/intel/mkl")
    endif()

    find_path(MKL_ROOT_DIR include/mkl.h PATHS ${MKLROOT_PATHS})
endif()

if(NOT MKL_ROOT_DIR)
    mkl_fail()
    return()
endif()

set(MKL_INCLUDE_DIRS ${MKL_ROOT_DIR}/include)
set(MKL_INCLUDE_HEADERS ${MKL_INCLUDE_DIRS}/mkl.h ${MKL_INCLUDE_DIRS}/mkl_version.h)

#determine arch
if(CMAKE_CXX_SIZEOF_DATA_PTR EQUAL 8)
    set(MKL_X64 1)
    set(MKL_ARCH "intel64")

    include(CheckTypeSize)
    CHECK_TYPE_SIZE(int _sizeof_int)
    if (_sizeof_int EQUAL 4)
        set(MKL_LP64 "lp64")
    else()
        set(MKL_LP64 "ilp64")
    endif()
else()
    set(MKL_ARCH "ia32")
endif()

if(MSVC)
    set(MKL_EXT ".lib")
    set(MKL_PRE "")
else()
    set(MKL_EXT ".a")
    set(MKL_PRE "lib")
endif()

set(MKL_LIB_DIR ${MKL_ROOT_DIR}/lib/${MKL_ARCH})
set(MKL_LIBRARIES ${MKL_LIB_DIR}/${MKL_PRE}mkl_core${MKL_EXT} ${MKL_LIB_DIR}/${MKL_PRE}mkl_intel_${MKL_LP64}${MKL_EXT})

if(MKL_WITH_TBB)
    list(APPEND MKL_LIBRARIES ${MKL_LIB_DIR}/${MKL_PRE}mkl_tbb_thread${MKL_EXT})
    list(APPEND MKL_LIBRARIES ${MKL_ROOT_DIR}/../tbb/lib/${MKL_ARCH}/tbb${MKL_EXT})
elseif(MKL_WITH_OPENMP)
    message(FATAL_ERROR "Multithreaded MKL is not supported yet")
else()
    list(APPEND MKL_LIBRARIES ${MKL_LIB_DIR}/${MKL_PRE}mkl_sequential${MKL_EXT})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL MKL_INCLUDE_HEADERS MKL_LIBRARIES)

if(MKL_FOUND)
    get_mkl_version(${MKL_INCLUDE_DIRS}/mkl_version.h)
    message(STATUS "Found MKL ${MKL_VERSION_STR} at: ${MKL_ROOT_DIR}")

    set(HAVE_MKL ON CACHE BOOL "True if MKL found")
    set(MKL_ROOT_DIR ${MKL_ROOT_DIR} CACHE PATH "Path to MKL directory")
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIRS} CACHE PATH "Path to MKL include directory")
    if(NOT UNIX)
        set(MKL_LIBRARIES ${MKL_LIBRARIES} CACHE FILEPATH "MKL libarries")
    else()
        #it's ugly but helps to avoid cyclic lib problem
        set(MKL_LIBRARIES ${MKL_LIBRARIES} ${MKL_LIBRARIES} ${MKL_LIBRARIES} "-lpthread" "-lm" "-ldl")
        set(MKL_LIBRARIES ${MKL_LIBRARIES} CACHE STRING "MKL libarries")
    endif()
else()

endif()