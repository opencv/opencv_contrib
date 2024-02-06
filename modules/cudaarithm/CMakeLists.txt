if(IOS OR WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudaarithm)
endif()

set(the_description "CUDA-accelerated Operations on Matrices")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow)

set(extra_dependencies "")
set(optional_dependencies "")
if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
  list(APPEND extra_dependencies CUDA::cudart_static CUDA::nppial${CUDA_LIB_EXT} CUDA::nppc${CUDA_LIB_EXT} CUDA::nppitc${CUDA_LIB_EXT} CUDA::nppig${CUDA_LIB_EXT} CUDA::nppist${CUDA_LIB_EXT} CUDA::nppidei${CUDA_LIB_EXT})
  if(HAVE_CUBLAS)
    list(APPEND optional_dependencies CUDA::cublas${CUDA_LIB_EXT})
    if(NOT CUDA_VERSION VERSION_LESS 10.1)
      list(APPEND optional_dependencies CUDA::cublasLt${CUDA_LIB_EXT})
    endif()
  endif()
  if(HAVE_CUFFT)
    # static version requires seperable compilation which is incompatible with opencv's current library structure
    list(APPEND optional_dependencies CUDA::cufft)
  endif()
else()
  if(HAVE_CUBLAS)
    list(APPEND optional_dependencies ${CUDA_cublas_LIBRARY})
  endif()
  if(HAVE_CUFFT)
    list(APPEND optional_dependencies ${CUDA_cufft_LIBRARY})
  endif()
endif()

ocv_add_module(cudaarithm opencv_core ${extra_dependencies} OPTIONAL opencv_cudev ${optional_dependencies} WRAP python)

ocv_module_include_directories()
ocv_glob_module_sources()

ocv_create_module()

ocv_add_accuracy_tests(DEPENDS_ON opencv_imgproc)
ocv_add_perf_tests(DEPENDS_ON opencv_imgproc)
