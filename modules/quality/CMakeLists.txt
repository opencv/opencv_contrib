if(CV_ICC AND NOT MSVC)
  ocv_module_disable(quality)
endif()

set(the_description "Quality API")

ocv_define_module(quality opencv_imgproc opencv_features2d WRAP python)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Woverloaded-virtual)
