if(HAVE_FASTCV)
  set(the_description "Qualcomm FastCV accelerated functions")
  ocv_define_module(fastcv opencv_core opencv_imgproc opencv_features2d opencv_video WRAP python java)
  ocv_module_include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${FastCV_INCLUDE_PATH})

  ocv_target_link_libraries(${the_module} ${FASTCV_LIBRARY})
  ocv_target_compile_definitions(${the_module} PRIVATE -DHAVE_FASTCV=1)
  ocv_install_3rdparty_licenses(FastCV "${OpenCV_BINARY_DIR}/3rdparty/fastcv/LICENSE")
else()
  ocv_module_disable(fastcv)
endif()
