set(the_description "RGBD algorithms")

find_package(Ceres QUIET)
ocv_define_module(rgbd opencv_core opencv_calib3d opencv_imgproc OPTIONAL opencv_viz WRAP python)
ocv_target_link_libraries(${the_module} ${CERES_LIBRARIES})

if(Ceres_FOUND)
  ocv_target_compile_definitions(${the_module} PUBLIC CERES_FOUND)
else()
  message(STATUS "CERES support is disabled. Ceres Solver is Required for Posegraph optimization")
endif()
