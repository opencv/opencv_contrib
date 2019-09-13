if(IOS OR WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudaoptflow)
endif()

set(the_description "CUDA-accelerated Optical Flow")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow -Wstrict-aliasing)

ocv_define_module(cudaoptflow opencv_video opencv_optflow opencv_cudaarithm opencv_cudawarping opencv_cudaimgproc OPTIONAL opencv_cudalegacy WRAP python)

set(NVIDIA_OPTICAL_FLOW_1_0_HEADERS_COMMIT "79c6cee80a2df9a196f20afd6b598a9810964c32")
set(NVIDIA_OPTICAL_FLOW_1_0_HEADERS_MD5 "ca5acedee6cb45d0ec610a6732de5c15")
set(NVIDIA_OPTICAL_FLOW_1_0_HEADERS_PATH "${OpenCV_BINARY_DIR}/3rdparty/NVIDIAOpticalFlowSDK_1_0_Headers")
ocv_download(FILENAME "${NVIDIA_OPTICAL_FLOW_1_0_HEADERS_COMMIT}.zip"
               HASH ${NVIDIA_OPTICAL_FLOW_1_0_HEADERS_MD5}
               URL
                 "https://github.com/NVIDIA/NVIDIAOpticalFlowSDK/archive/"
               DESTINATION_DIR "${NVIDIA_OPTICAL_FLOW_1_0_HEADERS_PATH}"
               STATUS NVIDIA_OPTICAL_FLOW_1_0_HEADERS_DOWNLOAD_SUCCESS
               ID "NVIDIA_OPTICAL_FLOW"
               UNPACK RELATIVE_URL)

if(NOT NVIDIA_OPTICAL_FLOW_1_0_HEADERS_DOWNLOAD_SUCCESS)
  message(STATUS "Failed to download NVIDIA_Optical_Flow_1_0 Headers")
else()
  add_definitions(-DHAVE_NVIDIA_OPTFLOW=1)
  ocv_include_directories(SYSTEM "${NVIDIA_OPTICAL_FLOW_1_0_HEADERS_PATH}/NVIDIAOpticalFlowSDK-${NVIDIA_OPTICAL_FLOW_1_0_HEADERS_COMMIT}")
endif()