if(IOS OR WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudaoptflow)
endif()

set(the_description "CUDA-accelerated Optical Flow")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow -Wstrict-aliasing)

ocv_define_module(cudaoptflow opencv_video opencv_optflow opencv_cudaarithm opencv_cudawarping opencv_cudaimgproc OPTIONAL opencv_cudalegacy WRAP python)

if(NOT CUDA_VERSION VERSION_LESS "10.0")
  set(NVIDIA_OPTICAL_FLOW_2_0_HEADERS_COMMIT "edb50da3cf849840d680249aa6dbef248ebce2ca")
  set(NVIDIA_OPTICAL_FLOW_2_0_HEADERS_MD5 "a73cd48b18dcc0cc8933b30796074191")
  set(NVIDIA_OPTICAL_FLOW_2_0_HEADERS_PATH "${OpenCV_BINARY_DIR}/3rdparty/NVIDIAOpticalFlowSDK_2_0_Headers")
  ocv_download(FILENAME "${NVIDIA_OPTICAL_FLOW_2_0_HEADERS_COMMIT}.zip"
                 HASH ${NVIDIA_OPTICAL_FLOW_2_0_HEADERS_MD5}
                 URL "https://github.com/NVIDIA/NVIDIAOpticalFlowSDK/archive/"
                 DESTINATION_DIR "${NVIDIA_OPTICAL_FLOW_2_0_HEADERS_PATH}"
                 STATUS NVIDIA_OPTICAL_FLOW_2_0_HEADERS_DOWNLOAD_SUCCESS
                 ID "NVIDIA_OPTICAL_FLOW"
                 UNPACK RELATIVE_URL)

  if(NOT NVIDIA_OPTICAL_FLOW_2_0_HEADERS_DOWNLOAD_SUCCESS)
    message(STATUS "Failed to download NVIDIA_Optical_Flow_2_0 Headers")
  else()
    message(STATUS "Building with NVIDIA Optical Flow API 2.0")
    add_definitions(-DHAVE_NVIDIA_OPTFLOW=2)
    ocv_include_directories(SYSTEM "${NVIDIA_OPTICAL_FLOW_2_0_HEADERS_PATH}/NVIDIAOpticalFlowSDK-${NVIDIA_OPTICAL_FLOW_2_0_HEADERS_COMMIT}")
  endif()
endif()