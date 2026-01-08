if(NOT HAVE_CUDA)
  ocv_module_disable(cudalegacy)
endif()

set(the_description "CUDA-accelerated Computer Vision (legacy)")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4130 /wd4324 /wd4512 /wd4310 -Wundef -Wmissing-declarations -Wuninitialized -Wshadow -Wdeprecated-declarations -Wstrict-aliasing -Wtautological-compare)

ocv_add_module(cudalegacy opencv_core opencv_video OPTIONAL opencv_objdetect opencv_imgproc opencv_calib3d opencv_cudaarithm opencv_cudafilters opencv_cudaimgproc)

ocv_module_include_directories()
ocv_glob_module_sources()

set(extra_libs "")
if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
  list(APPEND extra_libs CUDA::cudart${CUDA_LIB_EXT})
endif()
ocv_create_module(${extra_libs})

ocv_add_accuracy_tests(${extra_libs})
ocv_add_perf_tests(${extra_libs})
