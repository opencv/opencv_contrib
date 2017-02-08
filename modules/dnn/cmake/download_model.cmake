set(GoogleNet_url "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel")
set(GoogleNet_dst "$ENV{OPENCV_TEST_DATA_PATH}/dnn/bvlc_googlenet.caffemodel")
set(GoogleNet_sha "405fc5acd08a3bb12de8ee5e23a96bec22f08204")

set(VGG16_url "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel")
set(GG16_dst "$ENV{OPENCV_TEST_DATA_PATH}/dnn/VGG_ILSVRC_16_layers.caffemodel")

set(voc-fcn32s_url "http://dl.caffe.berkeleyvision.org/fcn32s-heavy-pascal.caffemodel")
set(voc-fcn32s_dst "$ENV{OPENCV_TEST_DATA_PATH}/dnn/fcn32s-heavy-pascal.caffemodel")

set(Alexnet_url "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel")
set(Alexnet_dst "$ENV{OPENCV_TEST_DATA_PATH}/dnn/bvlc_alexnet.caffemodel")
set(Alexnet_sha "9116a64c0fbe4459d18f4bb6b56d647b63920377")

set(Inception_url "https://github.com/petewarden/tf_ios_makefile_example/raw/master/data/tensorflow_inception_graph.pb")
set(Inception_dst "$ENV{OPENCV_TEST_DATA_PATH}/dnn/tensorflow_inception_graph.pb")

set(Enet_url "https://www.dropbox.com/sh/dywzk3gyb12hpe5/AABoUwqQGWvClUu27Z1EWeu9a/model-best.net?dl=0")
set(Enet_dst "$ENV{OPENCV_TEST_DATA_PATH}/dnn/Enet-model-best.net")

set(Fcn_url "http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel")
set(Fcn_dst "$ENV{OPENCV_TEST_DATA_PATH}/dnn/fcn8s-heavy-pascal.caffemodel")

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
