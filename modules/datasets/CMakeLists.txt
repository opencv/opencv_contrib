set(the_description "datasets framework")

set(filter_srcs "${CMAKE_CURRENT_LIST_DIR}/src/tinyxml2/tinyxml2.cpp")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  ocv_append_source_files_cxx_compiler_options(filter_srcs "-Wno-suggest-override")  # GCC
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  ocv_append_source_files_cxx_compiler_options(filter_srcs "-Wno-inconsistent-missing-override")  # Clang
endif()

ocv_define_module(datasets opencv_core opencv_imgcodecs opencv_ml opencv_flann OPTIONAL opencv_text WRAP python)

ocv_warnings_disable(CMAKE_CXX_FLAGS
    /wd4267                                      # flann, Win64
    -Wimplicit-fallthrough                       # tinyxml2.cpp
)
if(ARM AND CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  # suppress warnings from GCC only on 7.1 and later
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-psabi)
endif()
