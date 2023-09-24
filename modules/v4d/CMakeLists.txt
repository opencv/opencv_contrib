cmake_policy(SET CMP0079 NEW)

OCV_OPTION(OPENCV_V4D_ENABLE_ES3 "Enable OpenGL ES 3.0 backend for V4D" OFF
  VERIFY HAVE_OPENGL)

include(FetchContent)

if(NOT EMSCRIPTEN)
  find_package(glfw3 3 REQUIRED)
  find_package(OpenCL REQUIRED)
  include("FindOpenGL")
endif()

set(the_description "V4D Visualization Module")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)

# Check CXX Features
get_property(known_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
list (FIND known_features "cxx_std_20" idx)
if (${idx} LESS 0)
  message(STATUS "Module opencv_v4d disabled because it requires C++20")
  ocv_module_disable(v4d)
endif()

# Update submodules
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

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui/")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# Macro to download a file
macro(fetch_file download_name url hash)
  FetchContent_Declare(${download_name}
  URL ${url}
  URL_HASH SHA256=${hash}
  DOWNLOAD_NO_EXTRACT true
  TLS_VERIFY true
  )

  FetchContent_MakeAvailable(${download_name})
endmacro(fetch_file)

# Macro to add a WebAssembly sample
macro(add_emscripten_sample sample source models)
  if(NOT (TARGET ${sample}))
    ocv_add_executable(${sample} ${source})
    ocv_target_include_modules(${sample} opencv_core opencv_imgproc opencv_videoio opencv_video opencv_imgcodecs opencv_v4d opencv_face opencv_tracking opencv_objdetect opencv_stitching opencv_optflow opencv_imgcodecs opencv_features2d opencv_dnn  opencv_flann)
    ocv_target_link_libraries(${sample} opencv_v4d opencv_core opencv_imgproc opencv_videoio opencv_video opencv_imgcodecs opencv_v4d opencv_face opencv_tracking opencv_objdetect opencv_stitching opencv_optflow opencv_imgcodecs opencv_features2d opencv_dnn  opencv_flann)
    target_link_directories(${sample} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../lib")
    target_include_directories(${sample} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/v4d/detail/" "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui" "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui/backends/" "${CMAKE_CURRENT_SOURCE_DIR}/third/nanovg/src/")
    target_compile_features(${sample} PRIVATE cxx_std_20)
    set_target_properties(${sample} PROPERTIES SUFFIX ".js")
    add_custom_command(
        TARGET ${sample} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "${CMAKE_CURRENT_LIST_DIR}/samples/${sample}.html"
                "${CMAKE_CURRENT_BINARY_DIR}/../../bin/${sample}.html")
    if(${models})
      target_link_options(${sample} PRIVATE "SHELL:--preload-file ${CMAKE_CURRENT_BINARY_DIR}/assets/models/face_detection_yunet_2023mar.onnx")
      target_link_options(${sample} PRIVATE "SHELL:--preload-file ${CMAKE_CURRENT_BINARY_DIR}/assets/models/lbfmodel.yaml")
    endif()
  endif()
endmacro()

# Macro to add a native sample
macro(add_binary_sample sample source)
  if(NOT (TARGET ${sample}))
    ocv_add_executable(${sample} ${source})
  endif()
  ocv_target_link_libraries(${sample} OpenGL GLEW glfw X11)
  target_compile_features(${sample} PRIVATE cxx_std_20)
  target_link_directories(${sample} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../lib")
  target_include_directories(${sample} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/v4d/detail/" "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui" "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui/backends/" "${CMAKE_CURRENT_SOURCE_DIR}/third/nanovg/src/")
endmacro()

# Configure emscripten build flags
if(EMSCRIPTEN)
  set(EM_LINKER_FLAGS "-sENVIRONMENT=web,worker -sWASM_BIGINT=1 -sOFFSCREENCANVAS_SUPPORT -sOFFSCREENCANVASES_TO_PTHREAD=#v4dOffscreenCanvas -sEXPORTED_FUNCTIONS=_main,_v4dInitCapture -sEXPORTED_RUNTIME_METHODS=ccall -sPROXY_TO_PTHREAD=1 --use-preload-plugins --preload-file \"${CMAKE_CURRENT_BINARY_DIR}/assets/fonts/entypo.ttf\" --preload-file \"${CMAKE_CURRENT_BINARY_DIR}/assets/fonts/Roboto-Regular.ttf\" --preload-file \"${CMAKE_CURRENT_BINARY_DIR}/assets/fonts/Roboto-Bold.ttf\" --preload-file \"${CMAKE_CURRENT_BINARY_DIR}/doc/lena.png\" -sINITIAL_MEMORY=128MB -sALLOW_MEMORY_GROWTH=1 -sUSE_GLFW=3 -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2 --bind")

  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${EM_LINKER_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${EM_LINKER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EM_LINKER_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(EM_DEBUG_FLAGS "-s GL_DEBUG=1 -sSTACK_OVERFLOW_CHECK=2 -sASSERTIONS=2 -sNO_DISABLE_EXCEPTION_CATCHING -sEXCEPTION_DEBUG=1")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${EM_DEBUG_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${EM_DEBUG_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EM_DEBUG_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  endif()
endif()

#Configure NanoVG build options
if (NOT (TARGET nanovg))
  set(NANOVG_BUILD_EXAMPLES OFF)
  set(NANOVG_BUILD_GL2 OFF)
  if(OPENCV_V4D_ENABLE_ES3 OR EMSCRIPTEN)
    set(NANOVG_BUILD_GL3 OFF)
    set(NANOVG_BUILD_GLES3 ON)
  else()
    set(NANOVG_BUILD_GL3 ON)
    set(NANOVG_BUILD_GLES3 OFF)
  endif()
  set(NANOVG_BUILD_GLES2 OFF)
  set(NANOVG_BUILD_OUI OFF)
  set(NANOVG_BUILD_SVG OFF)
  set(NANOVG_SHARED_LIBS ON)
  set(NANOVG_USE_FREETYPE OFF)
  set(NANOVG_USE_STB ON)
  set(CMAKE_CXX_FLAGS "-Wno-error -pthread")
  set(CMAKE_C_FLAGS "-Wno-error -pthread")
  if(EMSCRIPTEN)
    set(CMAKE_CXX_FLAGS "-pthread")
    set(CMAKE_C_FLAGS "-pthread")
    set(NANOVG_STATIC_LIBS ON)
  else()
    set(NANOVG_STATIC_LIBS OFF)
  endif()
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third/nanovg/")
  target_include_directories(nanovg PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third/")
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/nanovg/src/")
  if(OPENCV_V4D_ENABLE_ES3)
    target_link_libraries(nanovg OpenGL::GLES3)
  elseif(NOT EMSCRIPTEN)
    target_link_libraries(nanovg OpenGL::OpenGL)
  endif()
endif()

# Add the opencv module
if(NOT (TARGET ${the_module}))
  ocv_add_module(v4d opencv_core opencv_imgproc opencv_videoio opencv_video)
  ocv_glob_module_sources("${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/" "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/v4d" "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/v4d/detail/")
  ocv_module_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/imgui" "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui/backends/" "${CMAKE_CURRENT_SOURCE_DIR}/third/nanovg/src/")
  ocv_create_module()
  ocv_add_samples(opencv_v4d opencv_core opencv_imgproc opencv_videoio opencv_video opencv_imgcodecs opencv_v4d opencv_face opencv_tracking opencv_objdetect opencv_stitching opencv_optflow opencv_imgcodecs opencv_features2d opencv_dnn opencv_flann)

  set_target_properties(${the_module} PROPERTIES LINKER_LANGUAGE CXX)

  # Add ImGui source files
  file(GLOB imgui_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui/*.cpp")
  file(GLOB imgui_backend_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui/backends/imgui_impl_opengl3*.cpp")
  file(GLOB imgui_glfw_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui/backends/imgui_impl_glfw.cpp")
  target_sources(${the_module} PUBLIC ${imgui_sources} ${imgui_backend_sources} ${imgui_glfw_sources})

  # Populate assets
  fetch_file("LBFMODEL" "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml" "70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b")

  fetch_file("YUNET" "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4")

  add_custom_command(TARGET ${the_module} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    "${CMAKE_CURRENT_BINARY_DIR}/assets")

  add_custom_command(TARGET ${the_module} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    "${CMAKE_CURRENT_BINARY_DIR}/assets/models")

  add_custom_command(TARGET ${the_module} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    "${CMAKE_CURRENT_BINARY_DIR}/assets/fonts")

  add_custom_command(TARGET ${the_module} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/samples/fonts/*.ttf"
    "${CMAKE_CURRENT_BINARY_DIR}/assets/fonts/")

  add_custom_command(TARGET ${the_module} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_LIST_DIR}/doc/lena.png"
    "${CMAKE_CURRENT_BINARY_DIR}/doc/lena.png")

  add_custom_command(TARGET ${the_module} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${lbfmodel_SOURCE_DIR}/lbfmodel.yaml"
    "${CMAKE_CURRENT_BINARY_DIR}/assets/models/")

  add_custom_command(TARGET ${the_module} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${yunet_SOURCE_DIR}/face_detection_yunet_2023mar.onnx"
    "${CMAKE_CURRENT_BINARY_DIR}/assets/models/")

  #Add sample targets
  if(BUILD_EXAMPLES)
    if(EMSCRIPTEN)
      add_emscripten_sample(example_v4d_display_image samples/display_image.cpp false)
      add_emscripten_sample(example_v4d_display_image_fb samples/display_image_fb.cpp false)
      add_emscripten_sample(example_v4d_vector_graphics samples/vector_graphics.cpp false)
      add_emscripten_sample(example_v4d_vector_graphics_and_fb samples/vector_graphics_and_fb.cpp false)
      add_emscripten_sample(example_v4d_render_opengl samples/render_opengl.cpp false)
      add_emscripten_sample(example_v4d_custom_source_and_sink samples/custom_source_and_sink.cpp false)
      add_emscripten_sample(example_v4d_font_rendering samples/font_rendering.cpp false)
      add_emscripten_sample(example_v4d_font_with_gui samples/font_with_gui.cpp false)
      add_emscripten_sample(example_v4d_video_editing samples/video_editing.cpp false)
      add_emscripten_sample(example_v4d_cube-demo samples/cube-demo.cpp false)
      add_emscripten_sample(example_v4d_many_cubes-demo samples/many_cubes-demo.cpp false)
      add_emscripten_sample(example_v4d_video-demo samples/video-demo.cpp false)
      add_emscripten_sample(example_v4d_nanovg-demo samples/nanovg-demo.cpp false)
      add_emscripten_sample(example_v4d_font-demo samples/font-demo.cpp false)
      add_emscripten_sample(example_v4d_shader-demo samples/shader-demo.cpp false)
      add_emscripten_sample(example_v4d_pedestrian-demo samples/pedestrian-demo.cpp false)
      add_emscripten_sample(example_v4d_optflow-demo samples/optflow-demo.cpp false)
      add_emscripten_sample(example_v4d_beauty-demo samples/beauty-demo.cpp true)
    else()
      add_binary_sample(example_v4d_display_image samples/display_image.cpp)
      add_binary_sample(example_v4d_display_image_fb samples/display_image_fb.cpp)
      add_binary_sample(example_v4d_vector_graphics samples/vector_graphics.cpp)
      add_binary_sample(example_v4d_vector_graphics_and_fb samples/vector_graphics_and_fb.cpp)
      add_binary_sample(example_v4d_render_opengl samples/render_opengl.cpp)
      add_binary_sample(example_v4d_custom_source_and_sink samples/custom_source_and_sink.cpp)
      add_binary_sample(example_v4d_font_rendering samples/font_rendering.cpp)
      add_binary_sample(example_v4d_font_with_gui samples/font_with_gui.cpp)
      add_binary_sample(example_v4d_video_editing samples/video_editing.cpp)
      add_binary_sample(example_v4d_cube-demo samples/cube-demo.cpp)
      add_binary_sample(example_v4d_many_cubes-demo samples/many_cubes-demo.cpp)
      add_binary_sample(example_v4d_video-demo samples/video-demo.cpp)
      add_binary_sample(example_v4d_nanovg-demo samples/nanovg-demo.cpp)
      add_binary_sample(example_v4d_font-demo samples/font-demo.cpp)
      add_binary_sample(example_v4d_shader-demo samples/shader-demo.cpp)
      add_binary_sample(example_v4d_pedestrian-demo samples/pedestrian-demo.cpp)
      add_binary_sample(example_v4d_optflow-demo samples/optflow-demo.cpp)
      add_binary_sample(example_v4d_beauty-demo samples/beauty-demo.cpp)
    endif()
  endif()

  if(OPENCV_V4D_ENABLE_ES3)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_V4D_USE_ES3=1")
  endif()


  target_compile_features(${the_module} PRIVATE cxx_std_20)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wdeprecated-enum-enum-conversion)
  target_link_directories(${the_module} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../lib")

  if(EMSCRIPTEN)
    ocv_target_link_libraries(${the_module} -lnanovg)
  else()
    ocv_target_link_libraries(${the_module} OpenCL OpenGL GLEW glfw -lnanovg)
  endif()
endif()
