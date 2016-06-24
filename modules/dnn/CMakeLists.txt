cmake_minimum_required(VERSION 2.8)

if(APPLE_FRAMEWORK OR WINRT)
  ocv_module_disable(dnn)
endif()

set(the_description "Deep neural network module. It allows to load models from different frameworks and to make forward pass")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)

ocv_add_module(dnn opencv_core opencv_imgproc WRAP python matlab)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-shadow -Wno-parentheses -Wmaybe-uninitialized -Wsign-promo -Wmissing-declarations -Wmissing-prototypes)
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4701)

# ----------------------------------------------------------------------------
# Resolve libprotobuf dependency
# ----------------------------------------------------------------------------
include(cmake/OpenCVFindLibProtobuf.cmake)
ocv_source_group("Src\\protobuf" FILES ${PROTOBUF_SRCS} ${PROTOBUF_HDRS})
ocv_module_include_directories(include ${PROTOBUF_INCLUDE_DIR})

# ----------------------------------------------------------------------------
# Try to find BLAS libraries
# ----------------------------------------------------------------------------
OCV_OPTION(${the_module}_WITH_BLAS "Use external BLAS library to speedup processing" OFF)
include(cmake/OpenCVFindCBLAS.cmake)

ocv_glob_module_sources(${PROTOBUF_SRCS} ${PROTOBUF_HDRS} ${CBLAS_H_PATH})
ocv_create_module(${PROTOBUF_LIBRARIES})
ocv_add_samples()
ocv_add_accuracy_tests()
ocv_add_perf_tests()

# ----------------------------------------------------------------------------
# Link BLAS
# ----------------------------------------------------------------------------
if(${the_module}_WITH_BLAS AND HAVE_BLAS)
    add_definitions(-DHAVE_CBLAS=1)
    ocv_module_include_directories(${${the_module}_BLAS_INCLUDE_DIR})
    ocv_add_dependencies(${the_module} ${${the_module}_BLAS_LIBRARIES})

    if(${the_module}_BLAS_BINARIES)
        ocv_install_target(${the_module} EXPORT ${the_module}_BLAS_BINARIES
                           RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT libs)
    endif()
else()
    add_definitions(-DHAVE_CBLAS=0)
endif()

# ----------------------------------------------------------------------------
# Download pre-trained models for complex testing on GoogLeNet and AlexNet
# ----------------------------------------------------------------------------
OCV_OPTION(${the_module}_DOWNLOAD_CAFFE_MODELS "Use GoogLeNet Caffe model for testing" OFF IF BUILD_TESTS AND PYTHON2_EXECUTABLE AND DEFINED ENV{OPENCV_TEST_DATA_PATH})
if(BUILD_TESTS AND PYTHON2_EXECUTABLE AND DEFINED ENV{OPENCV_TEST_DATA_PATH}
        AND (DOWNLOAD_EXTERNAL_TEST_DATA OR ${the_module}_DOWNLOAD_CAFFE_MODELS))
    add_custom_command( TARGET opencv_test_${name} POST_BUILD
                        COMMAND ${PYTHON2_EXECUTABLE} download_model.py test_models.json
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts )
    add_definitions(-DENABLE_CAFFE_MODEL_TESTS=1)
endif()

# ----------------------------------------------------------------------------
# Torch7 importer of blobs and models, produced by Torch.nn module
# ----------------------------------------------------------------------------
OCV_OPTION(${the_module}_BUILD_TORCH_IMPORTER "Build Torch model importer (experimental functionality!)" OFF)
if(${the_module}_BUILD_TORCH_IMPORTER)
    add_definitions(-DENABLE_TORCH_IMPORTER=1)
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4702 /wd4127 /wd4267) #supress warnings in original torch files
endif()

# ----------------------------------------------------------------------------
# Generating test data for Torch importer
# ----------------------------------------------------------------------------
OCV_OPTION(${the_module}_BUILD_TORCH_TESTS "Build Torch tests (installed torch7 with nn module is required)" ON IF BUILD_TESTS AND ${the_module}_BUILD_TORCH_IMPORTER)
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
