OCV_OPTION(WITH_ALSA "Include Advanced Linux Sound Architecture support" OFF
  VISIBLE_IF TRUE
  VERIFY HAVE_ALSA)
OCV_OPTION(WITH_SNDFILE "Include libsndfile library support" OFF
  VISIBLE_IF TRUE
  VERIFY HAVE_SNDFILE)

if(WITH_ALSA)
  include(cmake/FindALSA.cmake)
  message(STATUS "ALSA Found")
  set(HAVE_ALSA TRUE)
endif(WITH_ALSA)

if(WITH_SNDFILE)
  include(cmake/FindLIBSNDFILE.cmake)
  message(STATUS "LIBSNDFILE Found")
  set(HAVE_SNDFILE TRUE)
endif(WITH_SNDFILE)

ocv_add_module(audioio opencv_imgproc opencv_imgcodecs)

file(GLOB audioio_hdrs
  "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp")

file(GLOB audioio_src
  "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

ocv_set_module_sources(HEADERS ${audioio_hdrs} SOURCES ${audioio_src})
ocv_module_include_directories()
ocv_create_module()
ocv_add_accuracy_tests()
ocv_add_perf_tests()
