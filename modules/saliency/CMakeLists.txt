if(CV_ICC AND NOT MSVC)
  ocv_module_disable(saliency)
endif()

set(the_description "Saliency API")

ocv_define_module(saliency opencv_imgproc opencv_datasets opencv_features2d opencv_dnn opencv_ximgproc WRAP python)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Woverloaded-virtual)
