set(the_description "Object Detection")

ocv_define_module(dpm opencv_core opencv_imgproc opencv_objdetect OPTIONAL opencv_highgui WRAP python)

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4512) # disable warning on Win64
