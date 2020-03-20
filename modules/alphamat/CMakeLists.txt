if(NOT HAVE_EIGEN)
  set(DISABLE_MSG "Module opencv_alphamat disabled because the following dependencies are not found:")
  set(DISABLE_MSG "${DISABLE_MSG} Eigen")
  message(STATUS ${DISABLE_MSG})
  ocv_module_disable(alphamat)
endif()

cmake_minimum_required (VERSION 3.0)
project (myproject)
set (CMAKE_CXX_STANDARD 11)

ocv_define_module(alphamat opencv_core
    opencv_imgproc
    opencv_objdetect
    opencv_imgcodecs
    opencv_photo
)
