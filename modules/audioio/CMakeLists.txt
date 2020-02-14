OCV_OPTION(WITH_ALSA "Include Advanced Linux Sound Architecture support" OFF
  VISIBLE_IF TRUE
  VERIFY HAVE_ALSA)
OCV_OPTION(WITH_SNDFILE "Include libsndfile library support" OFF
  VISIBLE_IF TRUE
  VERIFY HAVE_SNDFILE)

ocv_add_module(audioio opencv_imgproc opencv_imgcodecs WRAP java python)

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
