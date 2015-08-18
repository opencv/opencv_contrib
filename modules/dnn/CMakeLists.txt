cmake_minimum_required(VERSION 2.8)

if(${CMAKE_PROJECT_NAME} STREQUAL "OpenCV")#build as OpenCV module

set(the_description "Deep neural network module. It allows to load models and to make forward pass")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)

OCV_OPTION(BUILD_LIBPROTOBUF_FROM_SOURCES "Build libprotobuf from sources or try to find it in system" OFF)
include(cmake/find_protobuf.cmake)

ocv_add_module(dnn opencv_imgproc opencv_core opencv_highgui WRAP python matlab)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-shadow -Wno-parentheses -Wmaybe-uninitialized -Wsign-promo -Wmissing-declarations -Wmissing-prototypes)
ocv_glob_module_sources(${PROTO_SRCS} ${PROTO_HDRS})
ocv_source_group("Src\\protobuf" FILES ${PROTO_SRCS} ${PROTO_HDRS})
ocv_module_include_directories(include ${PROTOBUF_INCLUDE_DIR})

ocv_create_module(${PROTOBUF_LIBRARIES})

ocv_add_samples()
ocv_add_accuracy_tests()
ocv_add_perf_tests()

OCV_OPTION(${the_module}_DOWNLOAD_CAFFE_MODELS "Use GoogLeNet Caffe model for testing" ON IF BUILD_TESTS AND PYTHON2_EXECUTABLE AND DEFINED ENV{OPENCV_TEST_DATA_PATH})
if(BUILD_TESTS AND ${the_module}_DOWNLOAD_CAFFE_MODELS)
    add_custom_command( TARGET opencv_test_${name} POST_BUILD
                        COMMAND ${PYTHON2_EXECUTABLE} download_model.py test_models.json
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts )
    add_definitions(-DENABLE_CAFFE_MODEL_TESTS=1)
endif()

OCV_OPTION(${the_module}_BUILD_TORCH_IMPORTER "Build Torch model importer" OFF)
if(${the_module}_BUILD_TORCH_IMPORTER)
    add_definitions(-DENABLE_TORCH_IMPORTER=1)
endif()

OCV_OPTION(${the_module}_BUILD_TORCH_TESTS "Build Torch tests (installed Torch7 with nn module is required)" ON IF BUILD_TESTS AND ${the_module}_BUILD_TORCH_IMPORTER)
if(${the_module}_BUILD_TORCH_TESTS)

    if(NOT DEFINED ENV{OPENCV_TEST_DATA_PATH})
        message(FATAL_ERROR "OPENCV_TEST_DATA_PATH environment variable was not specified")
    endif()

    execute_process(COMMAND th ${CMAKE_CURRENT_SOURCE_DIR}/testdata/dnn/torch/torch_nn_echo.lua RESULT_VARIABLE TORCH_STATUS)
    if(TORCH_STATUS)
        message(FATAL_ERROR "Torch executable \"th\" not found (status: ${TORCH_STATUS}) or nn module not found")
    endif()

    add_custom_command( TARGET opencv_test_${name} POST_BUILD
                        COMMAND th ${CMAKE_CURRENT_SOURCE_DIR}/testdata/dnn/torch/torch_gen_test_data.lua
                        WORKING_DIRECTORY  $ENV{OPENCV_TEST_DATA_PATH}/dnn/torch )
    add_definitions(-DENABLE_TORCH_TESTS=1)
endif()

else()#build as standalone module (for development purposes)

project(dnn_standalone)

include(cmake/find_protobuf.cmake)
find_package( OpenCV 3.0 REQUIRED )

file(GLOB_RECURSE MODULE_SOURCES "src/*.cpp")
file(GLOB_RECURSE MODULE_HEADERS "src/*.hpp" "src/*.h" "include/*.h*")
file(GLOB_RECURSE MODULE_TEST_SOURCES "test/*.cpp" "test/*.hpp" "test/*.h")

include_directories(include ${OpenCV_INCLUDE_DIRS})
add_library(_opencv_dnn SHARED ${MODULE_SOURCES} ${MODULE_HEADERS} ${PROTO_FILES} ${PROTO_HDRS})
add_executable(_opencv_test_dnn ${MODULE_TEST_SOURCES})
target_link_libraries(_opencv_dnn ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES})
target_link_libraries(_opencv_test_dnn _opencv_dnn opencv_ts)
target_include_directories(_opencv_dnn PRIVATE src src/caffe ${PROTOBUF_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

endif()
