if(NOT HAVE_VTK)
  ocv_module_disable(viz)
endif()

set(the_description "Viz")
include(${VTK_USE_FILE})

if(NOT BUILD_SHARED_LIBS)
  # We observed conflict between builtin 3rdparty libraries and
  # system-wide similar libraries (but with different versions) from VTK dependencies
  set(_conflicts "")
  foreach(dep ${VTK_LIBRARIES})
    if(("${dep}" MATCHES "libz\\." AND BUILD_ZLIB)
      OR ("${dep}" MATCHES "libjpeg\\." AND BUILD_JPEG)
      OR ("${dep}" MATCHES "libpng\\." AND BUILD_PNG)
      OR ("${dep}" MATCHES "libtiff\\." AND BUILD_TIFF)
    )
      list(APPEND _conflicts "${dep}")
    endif()
  endforeach()
  if(_conflicts)
    message(STATUS "Disabling VIZ module due to conflicts with VTK dependencies: ${_conflicts}")
    ocv_module_disable(viz)
  endif()
endif()

ocv_warnings_disable(CMAKE_CXX_FLAGS -Winconsistent-missing-override -Wsuggest-override)

ocv_add_module(viz opencv_core WRAP python)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

ocv_add_accuracy_tests()
ocv_add_perf_tests()
ocv_add_samples(opencv_imgproc opencv_calib3d opencv_features2d opencv_flann)

ocv_target_link_libraries(${the_module} PRIVATE ${VTK_LIBRARIES})

if(APPLE AND BUILD_opencv_viz)
  ocv_target_link_libraries(${the_module} PRIVATE "-framework Cocoa")
endif()

if(TARGET opencv_test_viz)
  set_target_properties(opencv_test_viz PROPERTIES MACOSX_BUNDLE TRUE)
endif()
