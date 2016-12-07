set(GoogleNet_url "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel")
set(GoogleNet_dst "$ENV{OPENCV_TEST_DATA_PATH}/dnn/bvlc_googlenet.caffemodel")
set(GoogleNet_sha "405fc5acd08a3bb12de8ee5e23a96bec22f08204")

set(VGG16_url "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel")
set(GG16_dst "$ENV{OPENCV_TEST_DATA_PATH}/dnn/VGG_ILSVRC_16_layers.caffemodel")

set(voc-fcn32s_url "http://dl.caffe.berkeleyvision.org/fcn32s-heavy-pascal.caffemodel")
set(voc-fcn32s_dst "$ENV{OPENCV_TEST_DATA_PATH}/dnn/fcn32s-heavy-pascal.caffemodel")

if(NOT model)
    set(model "GoogleNet")
endif()

message(STATUS "Downloading ${${model}_url} to ${${model}_dst}")

if(NOT EXISTS ${${model}_dst})
    if(${${model}_sha})
        file(DOWNLOAD ${${model}_url} ${${model}_dst} SHOW_PROGRESS EXPECTED_HASH SHA1=${${model}_sha} STATUS status_vec)
    else()
        file(DOWNLOAD ${${model}_url} ${${model}_dst} SHOW_PROGRESS STATUS status_vec)
    endif()

    list(GET status_vec 0 status)
    list(GET status_vec 1 status_msg)
    if(status EQUAL 0)
        message(STATUS "Ok! ${status_msg}")
    else()
        message(STATUS "Fail! ${status_msg}")
    endif()
endif()
