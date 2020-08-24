set(the_description "RGBD algorithms")

find_package(Ceres QUIET)

if(Ceres_FOUND)
  add_definitions("-DCERES_FOUND=1")
else()
  add_definitions("-DCERES_FOUND=0")
  message(STATUS "CERES support is disabled. Ceres Solver is Required for Posegraph optimization")
endif()

ocv_define_module(rgbd opencv_core opencv_calib3d opencv_imgproc OPTIONAL opencv_viz WRAP python)
ocv_target_link_libraries(${the_module} ${CERES_LIBRARIES})
