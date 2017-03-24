if(WINRT)
  ocv_module_disable(dnn)
endif()

set(the_description "Deep neural network module. It allows to load models from different frameworks and to make forward pass")

ocv_add_module(dnn opencv_core opencv_imgproc WRAP python matlab)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-shadow -Wno-parentheses -Wmaybe-uninitialized -Wsign-promo
                                     -Wmissing-declarations -Wmissing-prototypes
)
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4701 /wd4100)

if(MSVC)
  add_definitions( -D_CRT_SECURE_NO_WARNINGS=1 )
  ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4244 /wd4267 /wd4018 /wd4355 /wd4800 /wd4251 /wd4996 /wd4146
                                       /wd4305 /wd4127 /wd4100 /wd4512 /wd4125 /wd4389 /wd4510 /wd4610
                                       /wd4702 /wd4456 /wd4457 /wd4065 /wd4310 /wd4661 /wd4506
  )
else()
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-deprecated -Wmissing-prototypes -Wmissing-declarations -Wshadow
                                       -Wunused-parameter -Wunused-local-typedefs -Wsign-compare -Wsign-promo
                                       -Wundef -Wtautological-undefined-compare -Wignored-qualifiers -Wextra
                                       -Wunused-function -Wunused-const-variable -Wdeprecated-declarations
  )
endif()

if(APPLE_FRAMEWORK)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshorten-64-to-32)
endif()

if(ANDROID)
  add_definitions(-DDISABLE_POSIX_MEMALIGN -DTH_DISABLE_HEAP_TRACKING)
endif()

# ----------------------------------------------------------------------------
# Resolve libprotobuf dependency
# ----------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenCVFindLibProtobuf.cmake)
ocv_source_group("Src\\protobuf" FILES ${PROTOBUF_SRCS} ${PROTOBUF_HDRS})
ocv_module_include_directories(include ${PROTOBUF_INCLUDE_DIR})

ocv_glob_module_sources(${PROTOBUF_SRCS} ${PROTOBUF_HDRS} ${CBLAS_H_PROXY_PATH})
ocv_create_module(${PROTOBUF_LIBRARIES} ${LAPACK_LIBRARIES})
ocv_add_samples()
ocv_add_accuracy_tests()
ocv_add_perf_tests()

# ----------------------------------------------------------------------------
# Download pre-trained models for complex testing on GoogLeNet and AlexNet
# ----------------------------------------------------------------------------
OCV_OPTION(${the_module}_DOWNLOAD_MODELS "Use GoogLeNet Caffe model for testing" OFF IF BUILD_TESTS AND DEFINED ENV{OPENCV_TEST_DATA_PATH})
if(BUILD_TESTS AND DEFINED ENV{OPENCV_TEST_DATA_PATH} AND (DOWNLOAD_EXTERNAL_TEST_DATA OR ${the_module}_DOWNLOAD_MODELS))
    add_custom_command( TARGET opencv_test_${name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -Dmodel=GoogleNet -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_model.cmake)
    add_custom_command( TARGET opencv_test_${name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -Dmodel=Alexnet -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_model.cmake)
    add_custom_command( TARGET opencv_test_${name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -Dmodel=Inception -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_model.cmake)
    add_custom_command( TARGET opencv_test_${name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -Dmodel=Enet -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_model.cmake)
    add_definitions(-DENABLE_CAFFE_MODEL_TESTS=1)
    add_definitions(-DENABLE_CAFFE_ALEXNET_TEST=1)
    add_definitions(-DENABLE_TF_INCEPTION_TESTS=1)
    add_definitions(-DENABLE_TORCH_ENET_TESTS=1)
endif()

# ----------------------------------------------------------------------------
# Torch7 importer of blobs and models, produced by Torch.nn module
# ----------------------------------------------------------------------------
OCV_OPTION(${the_module}_BUILD_TORCH_IMPORTER "Build Torch model importer" ON)
if(${the_module}_BUILD_TORCH_IMPORTER)
    add_definitions(-DENABLE_TORCH_IMPORTER=1)
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4702 /wd4127 /wd4267) #supress warnings in original torch files

    if(NOT DEFINED HAVE_TORCH_EXE)
        execute_process(COMMAND th ${CMAKE_CURRENT_SOURCE_DIR}/testdata/dnn/torch/torch_nn_echo.lua RESULT_VARIABLE TORCH_EXE_STATUS)
        set(HAVE_TORCH_EXE OFF)
        if(${TORCH_EXE_STATUS} EQUAL 0)
            set(HAVE_TORCH_EXE ON)
        endif()
        set(HAVE_TORCH_EXE ${HAVE_TORCH_EXE} CACHE INTERNAL "Have torch binary")
    endif()
endif()

# ----------------------------------------------------------------------------
# Generating test data for Torch importer
# ----------------------------------------------------------------------------
OCV_OPTION(${the_module}_BUILD_TORCH_TESTS "Build Torch tests (installed torch7 with nn module is required)" ON IF BUILD_TESTS AND ${the_module}_BUILD_TORCH_IMPORTER AND HAVE_TORCH_EXE)
if(${the_module}_BUILD_TORCH_TESTS)

    if(NOT DEFINED ENV{OPENCV_TEST_DATA_PATH})
        message(FATAL_ERROR "OPENCV_TEST_DATA_PATH environment variable was not specified")
    endif()

    if(NOT HAVE_TORCH_EXE)
        message(FATAL_ERROR "Torch executable \"th\" not found or nn module not found")
    endif()

    add_custom_command( TARGET opencv_test_${name} POST_BUILD
                        COMMAND th ${CMAKE_CURRENT_SOURCE_DIR}/testdata/dnn/torch/torch_gen_test_data.lua
                        WORKING_DIRECTORY  $ENV{OPENCV_TEST_DATA_PATH}/dnn/torch )
    add_definitions(-DENABLE_TORCH_TESTS=1)
endif()
