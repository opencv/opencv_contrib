cmake_minimum_required (VERSION 3.0)
project (myproject)


set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

ocv_define_module(alphamat opencv_core
    opencv_imgproc
    opencv_objdetect
    opencv_imgcodecs
    opencv_photo     
)
