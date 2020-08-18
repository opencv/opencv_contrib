set(the_description "Text Detection and Recognition")
set(__extra_deps "")
if(DEBUG_opencv_text)
  list(APPEND __extra_deps PRIVATE_REQUIRED opencv_highgui)
endif()

ocv_define_module(text
    opencv_ml opencv_imgproc opencv_core opencv_features2d opencv_dnn
    ${__extra_deps}
    WRAP
      python
      java
      objc
)

if(TARGET ocv.3rdparty.tesseract)
  ocv_target_link_libraries(${the_module} LINK_PRIVATE ocv.3rdparty.tesseract)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/text_config.hpp.in
               ${CMAKE_BINARY_DIR}/text_config.hpp @ONLY)

ocv_include_directories(${CMAKE_CURRENT_BINARY_DIR})

ocv_add_testdata(samples/ contrib/text
    FILES_MATCHING PATTERN "*.xml" PATTERN "*.xml.gz" REGEX "scenetext[0-9]+.jpg"
)
