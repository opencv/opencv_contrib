set(the_description "Object Detection")

ocv_define_module(dpm opencv_core opencv_imgproc opencv_objdetect OPTIONAL opencv_highgui WRAP python)

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4512) # disable warning on Win64
if(ARM AND CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  # suppress warnings from GCC only on 7.1 and later
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-psabi)
endif()
