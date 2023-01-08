if(IOS OR APPLE OR WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudacodec)
endif()

set(the_description "CUDA-accelerated Video Encoding/Decoding")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wshadow)

set(required_dependencies opencv_core opencv_videoio opencv_cudaarithm opencv_cudawarping)
if(HAVE_NVCUVENC)
  list(APPEND required_dependencies opencv_cudaimgproc)
endif()

ocv_add_module(cudacodec ${required_dependencies} OPTIONAL opencv_cudev WRAP python)

ocv_module_include_directories()
ocv_glob_module_sources()

set(extra_libs "")

if(HAVE_NVCUVID OR HAVE_NVCUVENC)
  list(APPEND extra_libs ${CUDA_CUDA_LIBRARY})
  if(HAVE_NVCUVID)
    list(APPEND extra_libs ${CUDA_nvcuvid_LIBRARY})
  endif()
  if(HAVE_NVCUVENC)
      if(WIN32)
        list(APPEND extra_libs ${CUDA_nvencodeapi_LIBRARY})
      else()
        list(APPEND extra_libs ${CUDA_nvidia-encode_LIBRARY})
      endif()
  endif()
endif()

ocv_create_module(${extra_libs})

ocv_add_accuracy_tests()
ocv_add_perf_tests()