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
  list(APPEND extra_dependencies CUDA::cudart_static)

endif()

ocv_add_module(cudev opencv_core ${extra_dependencies} OPTIONAL ${optional_dependencies} WRAP python)

ocv_module_include_directories()
ocv_glob_module_sources()
ocv_create_module()


if(BUILD_TESTS AND NOT BUILD_opencv_world)
  add_subdirectory(test)
endif()
