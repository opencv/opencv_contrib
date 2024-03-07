set(the_description "RGBD algorithms")

ocv_define_module(rgbd opencv_core opencv_calib3d opencv_imgproc OPTIONAL opencv_viz WRAP python)

if(NOT HAVE_EIGEN)
  message(STATUS "rgbd: Eigen support is disabled. Eigen is Required for Posegraph optimization")
endif()

if(HAVE_OPENGL)
  ocv_target_link_libraries(${the_module} PRIVATE "${OPENGL_LIBRARIES}")
endif()
