include("FindGLEW")

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
set(CMAKE_CXX_FLAGS "${CMAKE_LD_FLAGS} -L../../lib")

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../.git")
# Update submodules as needed
  message(STATUS "Submodule update")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../"
                  RESULT_VARIABLE GIT_SUBMOD_RESULT)
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
  endif()
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui/ext/nanovg/src/")
include_directories(${GLEW_INCLUDE_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})

ocv_add_module(viz2d opencv_core opencv_imgproc opencv_videoio opencv_video)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

set(NANOGUI_BUILD_SHARED ON)
set(NANOGUI_BUILD_PYTHON OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui)

target_compile_features(opencv_viz2d PRIVATE cxx_std_20)
set_target_properties(opencv_viz2d PROPERTIES OUTPUT_NAME "viz2d")

#ocv_add_samples(opencv_aruco)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-parameter -Wdeprecated-enum-enum-conversion)
ocv_target_link_libraries(${the_module} ${GLEW_LIBRARIES} OpenCL -lnanovg -lnanogui -lGL)



