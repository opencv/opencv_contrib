if(NOT EMSCRIPTEN)
  find_package(glfw3 3 REQUIRED)
  include("FindGLEW")
  include("FindOpenGL")
else()
  pkg_check_modules (glew glew)
endif()

set(the_description "V4D Visualization Module")

macro(configure_files srcDir destDir)
    message(STATUS "Configuring directory ${destDir}")
    make_directory(${destDir})

    file(GLOB templateFiles RELATIVE ${srcDir} "${srcDir}/*")
    foreach(templateFile ${templateFiles})
        set(srcTemplatePath ${srcDir}/${templateFile})
        if(NOT IS_DIRECTORY ${srcTemplatePath})
            message(STATUS "Configuring file ${templateFile}")
            configure_file(
                    ${srcTemplatePath}
                    ${destDir}/${templateFile}
                    @ONLY)
        endif(NOT IS_DIRECTORY ${srcTemplatePath})
    endforeach(templateFile)
endmacro(configure_files)


macro(add_emscripten_sample sample source)
    ocv_add_executable(${sample} ${source})
    ocv_target_include_modules(${sample} opencv_core opencv_imgproc opencv_videoio opencv_video opencv_imgcodecs opencv_v4d opencv_face opencv_tracking opencv_objdetect opencv_stitching opencv_optflow opencv_imgcodecs opencv_features2d opencv_dnn  opencv_flann)
    ocv_target_link_libraries(${sample} LINK_PRIVATE opencv_core opencv_imgproc opencv_videoio opencv_video opencv_imgcodecs
    opencv_v4d opencv_face opencv_tracking opencv_objdetect opencv_stitching opencv_optflow opencv_imgcodecs opencv_features2d opencv_dnn  opencv_flann nanogui)
    target_link_directories(${sample} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../lib")
    target_compile_features(${sample} PRIVATE cxx_std_20)
    set_target_properties(${sample} PROPERTIES SUFFIX ".js")
    add_custom_command(
        TARGET ${sample} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "${CMAKE_CURRENT_LIST_DIR}/samples/${sample}.html"
                "${CMAKE_CURRENT_BINARY_DIR}/../../bin/${sample}.html")
endmacro()

macro(add_binary_sample sample)
    target_compile_features(${sample} PRIVATE cxx_std_20)
    target_link_directories(${sample} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../lib")
    target_include_directories(${sample} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/v4d/detail/")
    target_link_libraries(${sample} PRIVATE nanogui glfw OpenGL::GL GLEW::GLEW)
endmacro()

if(EMSCRIPTEN)
  set(EM_LINKER_FLAGS "-sPROXY_TO_PTHREAD=1 --use-preload-plugins --preload-file doc/lena.png -sINITIAL_MEMORY=128MB -sALLOW_MEMORY_GROWTH=1 -sUSE_GLFW=3 -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${EM_LINKER_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${EM_LINKER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EM_LINKER_FLAGS}")

  # if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  #   set(EM_DEBUG_FLAGS "-fsanitize=address -sSTACK_OVERFLOW_CHECK=2 -sASSERTIONS=2 -sNO_DISABLE_EXCEPTION_CATCHING -sEXCEPTION_DEBUG=1")
  #   set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${EM_DEBUG_FLAGS}")
  #   set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${EM_DEBUG_FLAGS}")
  #   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EM_DEBUG_FLAGS}")
  #   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
  # endif()
else()
  if(NOT GLEW_FOUND)
    message(STATUS "Module opencv_v4d disabled because GLEW was not found")
    ocv_module_disable(v4d)
  endif()
endif()

get_property(known_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
list (FIND known_features "cxx_std_20" idx)
if (${idx} LESS 0)
  message(STATUS "Module opencv_v4d disabled because it requires C++20")
  ocv_module_disable(v4d)
endif()

OCV_OPTION(OPENCV_V4D_ENABLE_ES3 "Enable OpenGL ES 3.0 backend for V4D" OFF
  VERIFY HAVE_OPENGL)

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

cmake_policy(SET CMP0079 NEW)

ocv_add_module(v4d opencv_core opencv_imgproc opencv_videoio opencv_video)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()
ocv_add_samples(opencv_core opencv_imgproc opencv_videoio opencv_video opencv_v4d opencv_face opencv_tracking opencv_objdetect opencv_stitching opencv_optflow opencv_imgcodecs opencv_features2d opencv_dnn opencv_flann)

add_custom_command(
  TARGET ${the_module} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_LIST_DIR}/doc/lena.png"
    "${CMAKE_CURRENT_BINARY_DIR}/doc/lena.png")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui/ext/glad/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui/ext/nanovg/src/")

if(BUILD_EXAMPLES)
  if(EMSCRIPTEN)
    set(NANOGUI_BUILD_GLFW OFF)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui/ext/glfw/include/")

    add_emscripten_sample(example_v4d_display_image samples/display_image.cpp)
    add_emscripten_sample(example_v4d_display_image_fb samples/display_image_fb.cpp)
    add_emscripten_sample(example_v4d_vector_graphics samples/vector_graphics.cpp)
    add_emscripten_sample(example_v4d_vector_graphics_and_fb samples/vector_graphics_and_fb.cpp)
    add_emscripten_sample(example_v4d_render_opengl samples/render_opengl.cpp)
    add_emscripten_sample(example_v4d_custom_source_and_sink samples/custom_source_and_sink.cpp)
    add_emscripten_sample(example_v4d_font_rendering samples/font_rendering.cpp)
    add_emscripten_sample(example_v4d_font_with_gui samples/font_with_gui.cpp)
    add_emscripten_sample(example_v4d_video_editing samples/video_editing.cpp)
    add_emscripten_sample(example_v4d_cube-demo samples/cube-demo.cpp)
    add_emscripten_sample(example_v4d_video-demo samples/video-demo.cpp)
    add_emscripten_sample(example_v4d_nanovg-demo samples/nanovg-demo.cpp)
    add_emscripten_sample(example_v4d_font-demo samples/font-demo.cpp)
    add_emscripten_sample(example_v4d_shader-demo samples/shader-demo.cpp)
    add_emscripten_sample(example_v4d_pedestrian-demo samples/pedestrian-demo.cpp)
    add_emscripten_sample(example_v4d_optflow-demo samples/optflow-demo.cpp)
    add_emscripten_sample(example_v4d_beauty-demo samples/beauty-demo.cpp)
  else()
    add_binary_sample(example_v4d_display_image)
    add_binary_sample(example_v4d_custom_source_and_sink)
    add_binary_sample(example_v4d_display_image_fb)
    add_binary_sample(example_v4d_font_rendering)
    add_binary_sample(example_v4d_font_with_gui)
    add_binary_sample(example_v4d_render_opengl)
    add_binary_sample(example_v4d_vector_graphics_and_fb)
    add_binary_sample(example_v4d_vector_graphics)
    add_binary_sample(example_v4d_video_editing)
    add_binary_sample(example_v4d_cube-demo)
    add_binary_sample(example_v4d_video-demo)
    add_binary_sample(example_v4d_nanovg-demo)
    add_binary_sample(example_v4d_font-demo)
    add_binary_sample(example_v4d_shader-demo)
    add_binary_sample(example_v4d_pedestrian-demo)
    add_binary_sample(example_v4d_optflow-demo)
    add_binary_sample(example_v4d_beauty-demo)
  endif()
endif()

if(EMSCRIPTEN)
  set(NANOGUI_BUILD_SHARED OFF)
  set(OPENCV_V4D_ENABLE_ES3 ON)
else()
  set(NANOGUI_BUILD_SHARED ON)
endif()
set(NANOGUI_BUILD_PYTHON OFF)
set(NANOGUI_BUILD_EXAMPLES OFF)
set(NANOGUI_BACKEND "OpenGL" CACHE STRING "Choose the backend used for rendering (OpenGL/GLES 2/GLES 3/Metal)" FORCE)

if(OPENCV_V4D_ENABLE_ES3)
    set(NANOGUI_BACKEND "GLES 3")
    set(NANOGUI_BUILD_GLAD OFF)
    set(NANOGUI_BUILD_GLFW OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_V4D_USE_ES3=1 -DNANOGUI_USE_GLES=1 -DNANOGUI_GLES_VERSION=3")
else()
  set(NANOGUI_BACKEND "OpenGL")
  set(NANOGUI_BUILD_GLAD ON)
  set(NANOGUI_BUILD_GLFW OFF)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui)
#add_library(nanovg STATIC "third/nanogui/ext/nanovg/src/nanovg.c")
target_compile_features(opencv_v4d PRIVATE cxx_std_20)
target_compile_features(nanogui PRIVATE cxx_std_20)
#target_compile_features(nanovg PRIVATE cxx_std_20)

target_link_libraries(nanogui PRIVATE glfw)
target_include_directories(nanogui PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui/ext/glad/include/")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wdeprecated-enum-enum-conversion)
target_link_directories(${the_module} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../lib")
target_include_directories(${the_module} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/v4d/detail/")
target_include_directories(${the_module} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third/nanogui/ext/glad/include/")
target_include_directories(${the_module} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}third/nanogui/ext/nanovg/src/")

if(EMSCRIPTEN)
  ocv_target_link_libraries(${the_module} -lnanogui)
else()
  ocv_target_link_libraries(${the_module} -lOpenCL -lnanogui glfw GLEW::GLEW OpenGL::GLU OpenGL::OpenGL)
endif()
