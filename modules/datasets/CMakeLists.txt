set(the_description "datasets framework")
ocv_define_module(datasets opencv_core opencv_face opencv_ml opencv_flann opencv_text WRAP python)

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4267) # flann, Win64
