set(the_description "Gestures recognition framework")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
find_package(Caffe)

if(CAFFE_FOUND)
    message(STATUS "Module ${the_description} is enabled.")
else()
    message(STATUS "Module ${the_description} is disabled (Caffe required).")
    return()
endif()

ocv_define_module(gestures opencv_core opencv_imgproc)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow -Wsign-compare)

include_directories(
    ${Caffe_INCLUDE_DIR}
)

target_link_libraries(
    opencv_gestures
    ${Caffe_LIBRARIES}
)

find_package(CUDA)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    add_definitions(-DCPU_ONLY)
endif()

if(BUILD_TESTS AND PYTHON2_EXECUTABLE AND DEFINED ENV{OPENCV_TEST_DATA_PATH})
    add_custom_command( TARGET opencv_test_${name} POST_BUILD
                        COMMAND ${PYTHON2_EXECUTABLE} download_extra.py dl_list.json
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts)
endif()
