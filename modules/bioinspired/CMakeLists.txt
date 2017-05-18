set(the_description "Biologically inspired algorithms")
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wdeprecated-register)

ocv_define_module(bioinspired opencv_core OPTIONAL opencv_highgui opencv_ocl WRAP java python)
