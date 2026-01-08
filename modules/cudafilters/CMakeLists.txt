if(IOS OR WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudafilters)
endif()

set(the_description "CUDA-accelerated Image Filtering")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow)

ocv_add_module(cudafilters opencv_imgproc opencv_cudaarithm WRAP python)

ocv_module_include_directories()
ocv_glob_module_sources()

set(build_libs "")
if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
  list(APPEND build_libs CUDA::cudart${CUDA_LIB_EXT} CUDA::nppif${CUDA_LIB_EXT} CUDA::nppim${CUDA_LIB_EXT})
endif()
ocv_create_module(${build_libs})

set(test_libs "")
if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
  list(APPEND test_libs CUDA::cudart${CUDA_LIB_EXT})
endif()
ocv_add_accuracy_tests(${test_libs})
ocv_add_perf_tests()
