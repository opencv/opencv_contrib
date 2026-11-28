cmake_policy(SET CMP0079 NEW)

OCV_OPTION(OPENCV_V4D_ENABLE_ES3 "Enable OpenGL ES 3.0 backend for V4D" OFF
  VERIFY HAVE_OPENGL)

include(FetchContent)

if(NOT EMSCRIPTEN)
  find_package(glfw3 3 REQUIRED)
  find_package(OpenCL REQUIRED)
  find_package(GLEW REQUIRED)
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

# Macro to add a native sample
macro(add_binary_sample sample source)
  if(NOT (TARGET ${sample}))
    ocv_add_executable(${sample} ${source})
  endif()
  ocv_target_link_libraries(${sample} OpenGL GLEW glfw X11 nanovg bgfx)
  target_compile_features(${sample} PRIVATE cxx_std_20)
  # set_property(TARGET ${sample} PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_link_directories(${sample} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../lib")
  target_include_directories(${sample} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/" "${CMAKE_CURRENT_SOURCE_DIR}/third/glad/include" "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui" "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui/backends/" "${CMAKE_CURRENT_SOURCE_DIR}/third/nanovg/src/" "${CMAKE_CURRENT_SOURCE_DIR}/third/bgfx.cmake/bgfx/include/" "${CMAKE_CURRENT_SOURCE_DIR}/third/bgfx.cmake/bx/include/" "${CMAKE_CURRENT_SOURCE_DIR}/third/bgfx.cmake/bimg/include/")
endmacro()
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
  # set(CMAKE_LD_FLAGS "${CMAKE_LqD_FLAGS} -fsanitize=address -static-libasan")

  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
  # set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -fsanitize=undefined -static-libasan")

  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
  # set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -fsanitize=thread -static-libasan")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-sign-promo")



if (NOT (TARGET nanovg))
    #Configure NanoVG build options
    if(OPENCV_V4D_ENABLE_ES3)
      add_definitions(-DNANOVG_GLES3=1 )
    else()
      add_definitions(-DNANOVG_GL3=1 )
    endif()

    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third/nanovg/")
    target_compile_options(nanovg PUBLIC -Wno-error)
    target_compile_options(nanovg PUBLIC -pthread)

    # # target_include_directories(nanovg PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third/nanovg/src/")
    # # include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/nanovg/src/")
    # if(OPENCV_V4D_ENABLE_ES3)
      # target_link_libraries(nanovg OpenGL::GLES3)
    # else()
      # target_link_libraries(nanovg OpenGL::OpenGL)
    # endif()
    # target_compile_features(nanovg PRIVATE cxx_std_20)

    install(TARGETS nanovg EXPORT OpenCVModules)
  endif()

  if (NOT (TARGET bgfx))
    set(BGFX_BUILD_EXAMPLES OFF)
    set(BGFX_LIBRARY_TYPE "SHARED")
    set(BGFX_INSTALL OFF)

  if(OPENCV_V4D_ENABLE_ES3)
        set(BGFX_OPENGLES_VERSION "30")
    else()
        set(BGFX_OPENGL_VERSION "32")
    endif()
  #-DBGFX_CONFIG_MULTITHREADED=0
    add_definitions(-DBGFX_CONFIG_PROFILER=0 -DBGFX_CONFIG_PASSIVE=1)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third/bgfx.cmake")


    target_compile_features(bgfx PRIVATE cxx_std_20)
    target_compile_options(bgfx PUBLIC -Wno-error)
    target_include_directories(bgfx PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third/glad/include")
    # target_link_libraries(bgfx PUBLIC glfw)
    install(TARGETS bgfx EXPORT OpenCVModules)
    install(TARGETS bimg EXPORT OpenCVModules)
    install(TARGETS bx EXPORT OpenCVModules)
  endif()
# Add the opencv module
if(NOT (TARGET ${the_module}))
  ocv_add_module(v4d opencv_core opencv_imgproc opencv_videoio opencv_video)
  file(GLOB imgui_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/imguicontext.cpp")
  file(GLOB imgui_backend_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui/backends/imgui_impl_opengl3*.cpp")
  file(GLOB imgui_glfw_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui/backends/imgui_impl_glfw.cpp")
  ocv_glob_module_sources("${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/" ${imgui_sources} ${imgui_backend_sources} ${imgui_glfw_sources})
  ocv_module_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/" "${CMAKE_CURRENT_SOURCE_DIR}/third/glad/include" "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui" "${CMAKE_CURRENT_SOURCE_DIR}/third/imgui/backends/" "${CMAKE_CURRENT_SOURCE_DIR}/third/nanovg/src/" "${CMAKE_CURRENT_SOURCE_DIR}/third/bgfx.cmake/bgfx/include/" "${CMAKE_CURRENT_SOURCE_DIR}/third/bgfx.cmake/bx/include/" "${CMAKE_CURRENT_SOURCE_DIR}/third/bgfx.cmake/bimg/include/")
  ocv_create_module()
  set_target_properties(${the_module} PROPERTIES LINKER_LANGUAGE CXX)

  ocv_add_samples(opencv_v4d opencv_core opencv_imgproc opencv_videoio opencv_video opencv_imgcodecs opencv_face opencv_tracking opencv_objdetect opencv_stitching opencv_optflow opencv_imgcodecs opencv_features2d opencv_dnn opencv_flann)
  # Populate assets
  fetch_file("LBFMODEL" "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml" "70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b")

  fetch_file("YUNET" "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4")

  add_custom_command(TARGET ${the_module} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    "${CMAKE_CURRENT_BINARY_DIR}/assets")

  add_custom_command(TARGET ${the_module} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    "${CMAKE_CURRENT_BINARY_DIR}/assets/doxygen")

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

  add_custom_command(TARGET ${the_module} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/third/doxygen-bootstrapped/customdoxygen.css"
    "${CMAKE_SOURCE_DIR}/doc/stylesheet.css")

  add_custom_command(TARGET ${the_module} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/third/doxygen-bootstrapped/example-site/header.html"
    "${CMAKE_SOURCE_DIR}/doc/")

  add_custom_command(TARGET ${the_module} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/third/doxygen-bootstrapped/example-site/footer.html"
    "${CMAKE_SOURCE_DIR}/doc/")

  list(APPEND CMAKE_DOXYGEN_HTML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/third/doxygen-bootstrapped/doxy-boot.js")

  #Add sample targets
  if(BUILD_EXAMPLES)
      add_binary_sample(example_v4d_display_image_fb samples/display_image_fb.cpp)
      add_binary_sample(example_v4d_display_image_nvg samples/display_image_nvg.cpp)
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
      add_binary_sample(example_v4d_bgfx-demo samples/bgfx-demo.cpp)
      add_binary_sample(example_v4d_bgfx-demo2 samples/bgfx-demo2.cpp)
      add_binary_sample(example_v4d_montage-demo samples/montage-demo.cpp)
  endif()

  if(OPENCV_V4D_ENABLE_ES3)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_V4D_USE_ES3=1")
  endif()


  target_compile_features(${the_module} PRIVATE cxx_std_20)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wdeprecated-enum-enum-conversion)
  target_link_directories(${the_module} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../lib")
  ocv_target_link_libraries(${the_module} OpenCL OpenGL::OpenGL glfw -lnanovg -lbgfx -lbimg -lbx)
endif()
