set(the_description "graphmodels framework")

ocv_add_module(graphmodels opencv_core)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-parameter -Wshadow -Wsign-compare)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF CACHE STRING "" FORCE)

SET(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++0x -mtune=native")
SET(CUDA_NVCC_FLAGS "-O3 --use_fast_math -gencode=arch=compute_20,code=sm_20
                                         -gencode=arch=compute_30,code=sm_30
                                         -gencode=arch=compute_35,code=sm_35")

SET(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -g")

# find OpenMP library
find_package(OpenMP REQUIRED)
IF(OPENMP_FOUND)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
message(STATUS "graphmodels: OpenMP found.")
ENDIF()

# find OPENBLAS library
FIND_PATH(OPENBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
include_directories(${OPENBLAS_INCLUDE_DIR})
FIND_LIBRARY(CBLAS_LIBRARY NAMES openblas)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CBLAS DEFAULT_MSG CBLAS_LIBRARY)
MARK_AS_ADVANCED(CBLAS_LIBRARY)

# find HDF5 library
find_package(HDF5 REQUIRED)
IF(HDF5_FOUND)
include_directories(${HDF5_INCLUDE_DIRS})
message(STATUS "graphmodels: HDF5 found." ${HDF5_DIR})
message(STATUS "graphmodels: HDF5 found." ${HDF5_LIBRARIES})
message(STATUS "graphmodels: HDF5 found." ${HDF5_INCLUDE_DIRS})
ENDIF()

# find Protobuf library
find_package(Protobuf REQUIRED)
IF(PROTOBUF_FOUND)
include_directories(${PROTOBUF_INCLUDE_DIRS})
message(STATUS "graphmodels: PROTOBUF found." ${PROTOBUF_DIR})
ENDIF()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROJECT_SOURCE_DIR}/proto/convnet_config.proto)

ocv_glob_module_sources()
ocv_module_include_directories()

#list(APPEND OPENCV_MODULE_${the_module}_SOURCES ${PROTO_SRCS})
CUDA_ADD_LIBRARY(${the_module} ${OPENCV_MODULE_${the_module}_SOURCES} ${PROTO_SRCS})
#ocv_create_module(${CUDA_cublas_LIBRARY} ${PROTOBUF_LIBRARIES} ${HDF5_LIBRARIES})
target_link_libraries(${the_module} ${CUDA_cublas_LIBRARY} ${PROTOBUF_LIBRARIES} ${HDF5_LIBRARIES})

ocv_add_samples()

