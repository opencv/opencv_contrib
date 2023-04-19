include("FindGLEW")
find_package(glm REQUIRED)

set(the_description "V4D Visualization Module")
if(NOT GLEW_FOUND)
  message(STATUS "Module opencv_v4d disabled because GLEW was not found")
  ocv_module_disable(v4d)
endif()

get_property(known_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
list (FIND known_features "cxx_std_20" idx)
if (${idx} LESS 0)
  message(STATUS "Module opencv_v4d disabled because it requires C++20")
  ocv_module_disable(v4d)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L../../lib")

OCV_OPTION(OPENCV_V4D_ENABLE_ES "Enable OpenGL ES backend for V4D" OFF
  VERIFY HAVE_OPENGL)
OCV_OPTION(OPENCV_V4D_ES3 "Use OpenGL ES3 instead of ES2" OFF
  VERIFY OPENCV_V4D_ENABLE_ES)


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

ocv_add_module(v4d opencv_core opencv_imgproc opencv_videoio opencv_video)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()
ocv_add_samples(opencv_face opencv_tracking opencv_objdetect opencv_stitching opencv_optflow opencv_imgcodecs opencv_features2d opencv_dnn)

if(BUILD_EXAMPLES)
  target_compile_features(example_v4d_display_image PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_display_image PRIVATE GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_custom_source_and_sink PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_custom_source_and_sink PRIVATE nanogui GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_display_image_fb PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_display_image_fb PRIVATE GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_font_rendering PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_font_rendering PRIVATE nanogui GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_font_with_gui PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_font_with_gui PRIVATE nanogui GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_render_opengl PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_render_opengl PRIVATE GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_vector_graphics_and_fb PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_vector_graphics_and_fb PRIVATE nanogui GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_vector_graphics PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_vector_graphics PRIVATE nanogui GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_video_editing PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_video_editing PRIVATE nanogui GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_cube-demo PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_cube-demo PRIVATE GLEW::glew OpenGL::GL glm::glm)
  target_compile_features(example_v4d_video-demo PRIVATE cxx_std_20 glm::glm)
  target_link_libraries(example_v4d_video-demo PRIVATE GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_nanovg-demo PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_nanovg-demo PRIVATE nanogui GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_font-demo PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_font-demo PRIVATE nanogui GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_shader-demo PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_shader-demo PRIVATE nanogui GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_pedestrian-demo PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_pedestrian-demo PRIVATE nanogui GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_optflow-demo PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_optflow-demo PRIVATE nanogui GLEW::glew OpenGL::GL)
  target_compile_features(example_v4d_beauty-demo PRIVATE cxx_std_20)
  target_link_libraries(example_v4d_beauty-demo PRIVATE nanogui GLEW::glew OpenGL::GL)
endif()

set(NANOGUI_BUILD_SHARED ON)
set(NANOGUI_BUILD_PYTHON OFF)
set(NANOGUI_BUILD_EXAMPLES OFF)
set(NANOGUI_BACKEND "OpenGL" CACHE STRING "Choose the backend used for rendering (OpenGL/GLES 2/GLES 3/Metal)" FORCE)

if(OPENCV_V4D_ENABLE_ES)
  if(OPENCV_V4D_ES3)
    set(NANOGUI_BACKEND "GLES 3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_V4D_ES_VERSION=3")
  else()
    set(NANOGUI_BACKEND "GLES 2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_V4D_ES_VERSION=2")
  endif()
else()
  set(NANOGUI_BACKEND "OpenGL")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui)

add_library(nanovg "third/nanogui/ext/nanovg/src/nanovg.c")
target_compile_features(opencv_v4d PRIVATE cxx_std_20)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-parameter -Wdeprecated-enum-enum-conversion -Wformat-truncation)
ocv_target_link_libraries(${the_module} ${GLEW_LIBRARIES} OpenCL -lnanovg -lnanogui -lGL)



