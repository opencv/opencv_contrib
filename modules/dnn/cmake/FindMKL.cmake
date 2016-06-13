# - Find the MKL libraries
# Modified from Armadillo's ARMA_FindMKL.cmake
# This module defines
#  MKL_INCLUDE_DIR, the directory for the MKL headers
#  MKL_LIB_DIR, the directory for the MKL library files
#  MKL_COMPILER_LIB_DIR, the directory for the MKL compiler library files
#  MKL_LIBRARIES, the libraries needed to use Intel's implementation of BLAS & LAPACK.
#  MKL_FOUND, If false, do not try to use MKL; if true, the macro definition USE_MKL is added.

# Set the include path
# TODO: what if MKL is not installed in /opt/intel/mkl?
# try to find at /opt/intel/mkl
# in windows, try to find MKL at C:/Program Files (x86)/Intel/Composer XE/mkl

if ( WIN32 )
  if(NOT DEFINED ENV{MKLROOT_PATH})
    #set(MKLROOT_PATH "C:/Program Files (x86)/Intel/Composer XE" CACHE PATH "Where the MKL are stored")
    set(MKLROOT_PATH "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows" CACHE PATH "Where the MKL are stored")
  endif(NOT DEFINED ENV{MKLROOT_PATH}) 
else ( WIN32 )
    set(MKLROOT_PATH "/opt/intel" CACHE PATH "Where the MKL are stored")
endif ( WIN32 )

if (EXISTS ${MKLROOT_PATH}/mkl)
    SET(MKL_FOUND TRUE)
    message("MKL is found at ${MKLROOT_PATH}/mkl")
    IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set( USE_MKL_64BIT On )
        if ( ARMADILLO_FOUND )
            if ( ARMADILLO_BLAS_LONG_LONG )
                set( USE_MKL_64BIT_LIB On )
                ADD_DEFINITIONS(-DMKL_ILP64)
                message("MKL is linked against ILP64 interface ... ")
            endif ( ARMADILLO_BLAS_LONG_LONG )
        endif ( ARMADILLO_FOUND )
    ELSE(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set( USE_MKL_64BIT Off )
    ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 8)
else (EXISTS ${MKLROOT_PATH}/mkl)
    SET(MKL_FOUND FALSE)
    message("MKL is NOT found ... ")
endif (EXISTS ${MKLROOT_PATH}/mkl)

if (MKL_FOUND)
    set(MKL_INCLUDE_DIR "${MKLROOT_PATH}/mkl/include")
    ADD_DEFINITIONS(-DUSE_MKL)
    if ( USE_MKL_64BIT )
        set(MKL_LIB_DIR "${MKLROOT_PATH}/mkl/lib/intel64")
        set(MKL_COMPILER_LIB_DIR "${MKLROOT_PATH}/compiler/lib/intel64")
        set(MKL_COMPILER_LIB_DIR ${MKL_COMPILER_LIB_DIR} "${MKLROOT_PATH}/lib/intel64")
        if ( USE_MKL_64BIT_LIB )
                if (WIN32)
                    set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_ilp64)
                else (WIN32)
                    set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_ilp64)
                endif (WIN32)
        else ( USE_MKL_64BIT_LIB )
                if (WIN32)
                    set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_lp64)
                else (WIN32)
                    set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_lp64)
                endif (WIN32)
        endif ( USE_MKL_64BIT_LIB )
    else ( USE_MKL_64BIT )
        set(MKL_LIB_DIR "${MKLROOT_PATH}/mkl/lib/ia32")
        set(MKL_COMPILER_LIB_DIR "${MKLROOT_PATH}/compiler/lib/ia32")
        set(MKL_COMPILER_LIB_DIR ${MKL_COMPILER_LIB_DIR} "${MKLROOT_PATH}/lib/ia32")
        if ( WIN32 )
            set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_c)
        else ( WIN32 )
            set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel)
        endif ( WIN32 )
    endif ( USE_MKL_64BIT )

    if (WIN32)
        SET(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_thread)
        SET(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_core)
        SET(MKL_LIBRARIES ${MKL_LIBRARIES} libiomp5md)
    else (WIN32)
        SET(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_gnu_thread)
        SET(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_core)
    endif (WIN32) 
endif (MKL_FOUND)

IF (MKL_FOUND)
    IF (NOT MKL_FIND_QUIETLY)
        MESSAGE(STATUS "Found MKL libraries: ${MKL_LIBRARIES}")
        MESSAGE(STATUS "MKL_INCLUDE_DIR: ${MKL_INCLUDE_DIR}")
        MESSAGE(STATUS "MKL_LIB_DIR: ${MKL_LIB_DIR}")
        MESSAGE(STATUS "MKL_COMPILER_LIB_DIR: ${MKL_COMPILER_LIB_DIR}")
    ENDIF (NOT MKL_FIND_QUIETLY)

    INCLUDE_DIRECTORIES( ${MKL_INCLUDE_DIR} )
    LINK_DIRECTORIES( ${MKL_LIB_DIR} ${MKL_COMPILER_LIB_DIR} )
ELSE (MKL_FOUND)
    IF (MKL_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find MKL libraries")
    ENDIF (MKL_FIND_REQUIRED)
ENDIF (MKL_FOUND)

# MARK_AS_ADVANCED(MKL_LIBRARY)