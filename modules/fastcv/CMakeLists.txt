set(the_description "FastCV extension module")
ocv_define_module(fastcv opencv_core opencv_imgproc opencv_features2d opencv_video WRAP python java)

if (FASTCV_DIR)
  if(NOT ARM AND NOT AARCH64)
    # the reason this warning is not fatal is to enable proper code analysis
    # in your favourite IDE
    message(WARNING "FastCV supports ARM platform only!")
  endif()

  # here you can change the library to a specific one
  set(FASTCV_LIB "/libs/Android/lib64/libfastcvopt.so")


  ocv_module_include_directories("${FASTCV_DIR}/inc")
  ocv_target_link_libraries(${the_module} LINK_PRIVATE "${FASTCV_DIR}/${FASTCV_LIB}")
  ocv_target_compile_definitions(${the_module} PRIVATE FAST_CV_FOUND)
endif()
