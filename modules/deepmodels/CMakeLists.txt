set(the_description "deepmodels framework")

ocv_add_module(deepmodels opencv_core)
ocv_glob_module_sources()
ocv_module_include_directories()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
SET(CUDA_NVCC_FLAGS "-O3 --use_fast_math -gencode=arch=compute_20,code=sm_20
                                         -gencode=arch=compute_30,code=sm_30
                                         -gencode=arch=compute_35,code=sm_35")

SET(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -g")

CUDA_ADD_LIBRARY(${the_module} ${OPENCV_MODULE_${the_module}_SOURCES})
CUDA_ADD_CUBLAS_TO_TARGET( ${the_module} )

ocv_add_samples()

