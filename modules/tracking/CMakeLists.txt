set(the_description "Tracking API")
ocv_define_module(tracking opencv_imgproc opencv_core opencv_video opencv_plot OPTIONAL opencv_dnn opencv_datasets WRAP java python)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-shadow /wd4458)
if(ARM AND CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  # suppress warnings from GCC only on 7.1 and later
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-psabi)
endif()
