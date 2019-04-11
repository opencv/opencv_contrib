set(the_description "Text Detection and Recognition")
ocv_define_module(text opencv_ml opencv_imgproc opencv_core opencv_features2d opencv_dnn OPTIONAL opencv_highgui WRAP python java)

if(NOT CMAKE_CROSSCOMPILING OR OPENCV_FIND_TESSERACT)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  find_package(Tesseract QUIET)
  find_package(Leptonica QUIET)

  if(Leptonica_FOUND)
    set(Lept_LIBRARY ${Leptonica_LIBRARIES})
  else()
    find_library(Lept_LIBRARY NAMES lept
      HINTS
      /usr/lib
      /usr/local/lib)
  endif()
  if(Lept_LIBRARY)
    message(STATUS "Found Tesseract dependencies: Leptonica ${Leptonica_VERSION}")
  endif()

  if(Tesseract_FOUND)
    set(Tesseract_LIBRARY ${Tesseract_LIBRARIES})
    set(Tesseract_INCLUDE_DIR ${Tesseract_INCLUDE_DIRS})
  else()
    find_path(Tesseract_INCLUDE_DIR tesseract/baseapi.h
      HINTS
      /usr/include
      /usr/local/include)

    find_library(Tesseract_LIBRARY NAMES tesseract
      HINTS
      /usr/lib
      /usr/local/lib)
  endif()
 
  if(Tesseract_INCLUDE_DIR AND Tesseract_LIBRARY AND Lept_LIBRARY)
    message(STATUS "Tesseract:   YES ${Tesseract_VERSION}")
    set(HAVE_TESSERACT 1)
 
    ocv_include_directories(${Tesseract_INCLUDE_DIR})
    ocv_target_link_libraries(${the_module} ${Tesseract_LIBRARY})
    ocv_target_link_libraries(${the_module} ${Lept_LIBRARY})
  else()
    message(STATUS "Tesseract:   NO")
  endif()
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/text_config.hpp.in
               ${CMAKE_BINARY_DIR}/text_config.hpp @ONLY)

ocv_include_directories(${CMAKE_CURRENT_BINARY_DIR})

ocv_add_testdata(samples/ contrib/text
    FILES_MATCHING PATTERN "*.xml" PATTERN "*.xml.gz" REGEX "scenetext[0-9]+.jpg"
)
