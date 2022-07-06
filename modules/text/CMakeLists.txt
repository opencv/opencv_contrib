set(the_description "Text Detection and Recognition")
set(__extra_deps "")
if(DEBUG_opencv_text)
  list(APPEND __extra_deps PRIVATE_REQUIRED opencv_highgui)
endif()
if(ARM AND CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  # suppress warnings from GCC only on 7.1 and later
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-psabi)
endif()

ocv_define_module(text
    opencv_ml opencv_imgproc opencv_core opencv_features2d opencv_dnn
    ${__extra_deps}
    WRAP
      python
      java
)

if(HAVE_TESSERACT)
  ocv_include_directories(${Tesseract_INCLUDE_DIRS})
  ocv_target_link_libraries(${the_module} ${Tesseract_LIBRARIES})
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/text_config.hpp.in
               ${CMAKE_BINARY_DIR}/text_config.hpp @ONLY)

ocv_include_directories(${CMAKE_CURRENT_BINARY_DIR})

ocv_add_testdata(samples/ contrib/text
    FILES_MATCHING PATTERN "*.xml" PATTERN "*.xml.gz" REGEX "scenetext[0-9]+.jpg"
)
