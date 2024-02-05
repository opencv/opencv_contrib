if(IOS OR WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudawarping)
endif()

set(the_description "CUDA-accelerated Image Warping")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow)
set(extra_libs "")
if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
  set(extra_libs CUDA::nppial${CUDA_LIB_EXT} CUDA::nppig${CUDA_LIB_EXT})
endif()
ocv_define_module(cudawarping opencv_core opencv_imgproc ${extra_libs} OPTIONAL opencv_cudev WRAP python)
