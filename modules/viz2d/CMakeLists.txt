include("FindGLEW")
include("FindPkgConfig")

set(the_description "Viz2D Visualization Module")

if(NOT GLEW_FOUND)
  message(STATUS "Module opencv_viz2d disabled because GLEW was not found")
  ocv_module_disable(viz2d)
endif()

get_property(known_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
list (FIND known_features "cxx_std_20" idx)
if (${idx} LESS 0)
  message(STATUS "Module opencv_viz2d disabled because it requires C++20")
  ocv_module_disable(viz2d)
endif()

#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVIZ2D_USE_ES3=1")

include_directories(${GLEW_INCLUDE_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})

ocv_add_module(viz2d opencv_core opencv_imgproc opencv_videoio opencv_video)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

#ocv_add_samples(opencv_aruco)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-parameter)
message(STATUS "viz2d: ${nangui_LIBRARIES}")
ocv_target_link_libraries(${the_module} ${GLEW_LIBRARIES} OpenCL)
target_compile_features(opencv_viz2d PRIVATE cxx_std_20)
set_target_properties(opencv_viz2d PROPERTIES OUTPUT_NAME "viz2d")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui)
