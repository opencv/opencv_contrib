include("FindGLEW")
include("FindPkgConfig")

set(the_description "Viz2D Visualization Module")

find_package(nanogui CONFIG)
find_package(glfw3 CONFIG)
pkg_check_modules(nanovg nanovg)

if(NOT glfw3_FOUND)
  message(STATUS "Module opencv_viz2d disabled because GLFW3 was not found")
  ocv_module_disable(viz2d)
elseif(NOT GLEW_FOUND)
  message(STATUS "Module opencv_viz2d disabled because GLEW was not found")
  ocv_module_disable(viz2d)
elseif(NOT nanovg_FOUND)
  message(STATUS "Module opencv_viz2d disabled because NanoVG was not found")
  ocv_module_disable(viz2d)
elseif(NOT nanogui_FOUND)
  message(STATUS "Module opencv_viz2d disabled because NanoGUI was not found")
  ocv_module_disable(viz2d)
endif()

get_property(known_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
list (FIND known_features "cxx_std_20" idx)
if (${idx} LESS 0)
  message(STATUS "Module opencv_viz2d disabled because it requires C++20")
  ocv_module_disable(viz2d)
endif()

#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVIZ2D_USE_ES3=1")

find_package(Git QUIET)
if(GIT_FOUND)
# Update submodules as needed
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/third/nanovg/CMakeLists.txt" AND NOT EXISTS "${PROJECT_SOURCE_DIR}/third/nanogui/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

add_subdirectory(third/nanovg)
add_subdirectory(third/nanogui)

include_directories(${glfw3_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${nanovg_INCLUDE_DIRS})
include_directories(${nanogui_INCLUDE_DIRS})
link_directories(${glfw3_LIBRARY_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${nanovg_LIBRARY_DIRS})
link_directories(${nanogui_LIBRARY_DIRS})

ocv_add_module(viz2d opencv_core opencv_imgproc opencv_videoio opencv_video)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

#ocv_add_samples(opencv_aruco)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-parameter)
message(STATUS "viz2d: ${nangui_LIBRARIES}")
#For some reasone glfw3_LIBRARIES is empty although it is found by find_package
ocv_target_link_libraries(${the_module} glfw ${GLEW_LIBRARIES} ${nanovg_LIBRARIES} nanogui OpenCL)
target_compile_features(opencv_viz2d PRIVATE cxx_std_20)
