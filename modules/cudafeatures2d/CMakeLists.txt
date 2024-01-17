if(IOS OR WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudafeatures2d)
endif()

set(the_description "CUDA-accelerated Feature Detection and Description")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4100 /wd4324 /wd4512 /wd4515 -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter -Wshadow)
if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
  ocv_module_include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()
ocv_define_module(cudafeatures2d opencv_features2d opencv_cudafilters opencv_cudawarping WRAP python)
