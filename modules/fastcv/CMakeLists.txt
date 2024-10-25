if((ARM OR AARCH64) AND (ANDROID OR (UNIX AND NOT APPLE AND NOT IOS AND NOT XROS)))
  set(the_description       "FastCV extension module")
  set(FCV_MODULE_HEADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/inc")
  set(FCV_MODULE_LIB_DIR    "${CMAKE_CURRENT_BINARY_DIR}/libs")

  # here you can change the library to a specific one
  set(FASTCV_LIB "${FCV_MODULE_LIB_DIR}/libfastcvopt.so")

  if((NOT EXISTS ${FCV_MODULE_HEADER_DIR}) OR (NOT EXISTS ${FCV_MODULE_LIB_DIR}))
    include("${CMAKE_CURRENT_SOURCE_DIR}/fastcv.cmake")
    download_fastcv("${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  ocv_define_module(fastcv opencv_core opencv_imgproc opencv_features2d opencv_video WRAP python java)
  ocv_module_include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${FCV_MODULE_HEADER_DIR})
  ocv_target_link_libraries(${the_module} ${FASTCV_LIB})
  ocv_target_compile_definitions(${the_module} PRIVATE FAST_CV_FOUND)
else()
  message(FATAL_ERROR "FastCV supports ARM Platform on Android and Linux only!")
endif()
