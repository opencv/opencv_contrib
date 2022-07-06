set(the_description "Object Detection using CNNs")
if(ARM AND CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  # suppress warnings from GCC only on 7.1 and later
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-psabi)
endif()
ocv_define_module(dnn_objdetect opencv_core opencv_imgproc opencv_dnn
    OPTIONAL opencv_highgui opencv_imgcodecs  # samples
)
