if(IOS OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudabgsegm)
endif()

set(the_description "CUDA-accelerated Background Segmentation")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow)
if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
  ocv_module_include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()
ocv_define_module(cudabgsegm opencv_video WRAP python)
