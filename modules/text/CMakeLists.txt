set(the_description "Text Detection and Recognition")
ocv_define_module(text opencv_ml opencv_highgui opencv_imgproc opencv_core opencv_features2d WRAP python)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Tesseract)
if(Tesseract_FOUND)
    message(STATUS "Tesseract:   YES")
    set(HAVE_TESSERACT 1)
else()
    message(STATUS "Tesseract:   NO")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/text_config.hpp.in
               ${CMAKE_BINARY_DIR}/text_config.hpp @ONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(${Tesseract_FOUND})
include_directories(${Tesseract_INCLUDE_DIR})
endif()

if(${Tesseract_FOUND})
  target_link_libraries(opencv_text ${Tesseract_LIBS})
endif()
