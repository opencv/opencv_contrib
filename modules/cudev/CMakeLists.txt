if(NOT HAVE_CUDA)
  ocv_module_disable(cudev)
endif()

set(the_description "CUDA device layer")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4189 /wd4505 -Wundef -Wmissing-declarations -Wunused-function -Wunused-variable -Wenum-compare -Wshadow)

set(extra_dependencies "")
set(optional_dependencies "")

if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
  if(UNIX AND NOT BUILD_SHARED_LIBS AND CUDA_VERSION_STRING VERSION_GREATER_EQUAL 9.2 AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    set(CUDA_FFT_LIB_EXT "_static_nocallback")
  endif()
  list(APPEND extra_dependencies CUDA::cudart_static CUDA::nppial${CUDA_LIB_EXT} CUDA::nppc${CUDA_LIB_EXT} CUDA::nppitc${CUDA_LIB_EXT} CUDA::nppig${CUDA_LIB_EXT} CUDA::nppist${CUDA_LIB_EXT} CUDA::nppidei${CUDA_LIB_EXT})
  if(HAVE_CUBLAS)
    list(APPEND optional_dependencies CUDA::cublas${CUDA_LIB_EXT})
    if(NOT CUDA_VERSION VERSION_LESS 10.1)
      list(APPEND optional_dependencies CUDA::cublasLt${CUDA_LIB_EXT})
    endif()
  endif()
  if(HAVE_CUFFT)
    # static version requires seperable compilation which is incompatible with opencv's current library structure
    # the cufft_static_nocallback variant does not requires seperable compilation. callbacks are currently not used.
    list(APPEND optional_dependencies CUDA::cufft${CUDA_FFT_LIB_EXT})
  endif()
else()
  if(HAVE_CUBLAS)
    list(APPEND optional_dependencies ${CUDA_cublas_LIBRARY})
  endif()
  if(HAVE_CUFFT)
    list(APPEND optional_dependencies ${CUDA_cufft_LIBRARY})
  endif()
endif()

ocv_add_module(cudev opencv_core ${extra_dependencies} OPTIONAL opencv_cudev ${optional_dependencies} WRAP python)

ocv_module_include_directories()
ocv_glob_module_sources()
ocv_create_module()


if(BUILD_TESTS AND NOT BUILD_opencv_world)
  add_subdirectory(test)
endif()
