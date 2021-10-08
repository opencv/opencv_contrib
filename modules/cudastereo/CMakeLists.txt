if(IOS OR  WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudastereo)
endif()

set(the_description "CUDA-accelerated Stereo Correspondence")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow)

ocv_define_module(cudastereo opencv_calib3d OPTIONAL opencv_cudev opencv_cudaarithm opencv_cudaimgproc WRAP python)

if(HAVE_NVIDIA_OPTICAL_FLOW_SDK)
  add_definitions(-DHAVE_NVIDIA_OPTFLOW=${NVIDIA_OPTICAL_FLOW_SDK_MAJOR_VERSION})
  ocv_include_directories(SYSTEM "${NVIDIA_OPTICAL_FLOW_SDK_INCLUDE}")
  ocv_target_link_libraries(${the_module} PRIVATE cuda)
endif()
