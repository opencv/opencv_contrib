if(NOT HAVE_EIGEN)
  message(STATUS "Module opencv_alphamat disabled because the following dependencies are not found: Eigen")
  ocv_module_disable(alphamat)
endif()

ocv_define_module(alphamat
    opencv_core
    opencv_imgproc
    WRAP python
)
