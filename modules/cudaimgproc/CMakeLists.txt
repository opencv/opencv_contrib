if(IOS OR WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudaimgproc)
endif()

set(the_description "CUDA-accelerated Image Processing")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4100 /wd4324 /wd4512 /wd4515 -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter)

ocv_define_module(cudaimgproc opencv_imgproc OPTIONAL opencv_cudev opencv_cudaarithm opencv_cudafilters WRAP python)

if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
  ocv_target_link_libraries(${the_module} PRIVATE CUDA::cudart${CUDA_LIB_EXT} CUDA::nppial${CUDA_LIB_EXT} CUDA::nppist${CUDA_LIB_EXT} CUDA::nppicc${CUDA_LIB_EXT} CUDA::nppidei${CUDA_LIB_EXT})
endif()
