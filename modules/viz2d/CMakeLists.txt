include("FindGLEW")
set(the_description "Viz2D Visualization Module")

find_package(glfw3 CONFIG)

if(NOT glfw3_FOUND)
  message(STATUS "Module opencv_viz2d disabled because GLFW3 was not found")
  ocv_module_disable(viz2d)
elseif(NOT GLEW_FOUND)
  message(STATUS "Module opencv_viz2d disabled because GLEW was not found")
  ocv_module_disable(viz2d)
else() # we need C++20
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qstd=c++20")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
  endif()
endif()

include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
link_directories(${GLFW_LIBRARY_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})

ocv_add_module(viz2d opencv_core opencv_imgproc opencv_videoio opencv_video)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

#ocv_add_samples(opencv_aruco)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-parameter)
ocv_target_link_libraries(${the_module} ${GLFW_LIBRARIES} ${GLEW_LIBRARIES})
