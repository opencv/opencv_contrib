### Gestures Recognition framework

This module provides algorithm framework to classify gestures based on multi-modalities input : color, depth and mocap stream.

#### Reference

The current implementation is based on the deep neural network architecture described in:

    "'Natalia Neverova, Christian Wolf, Graham W. Taylor and Florian Nebout.
    ModDrop: adaptive multi-modal gesture recognition.
    To appear in IEEE Transactions on Pattern Analysis and Machine Intelligence".

An arxiv version is available at:
    http://arxiv.org/abs/1501.00102.

It corresponds to our submission to the CVPR-2015 Vision-Challenge in categorie « Gesture Recognition ».

#### Caffe Dependency installation

Deep Neural Network in this module are implemented using the [Caffe](http://caffe.berkeleyvision.org) library.

The support of some required 3D operations has been added on the `nd-pooling-public` branch of our fork of the Caffe repository:
https://github.com/awabot-dev/caffe.git

Detailled installation instructions can be found here: http://caffe.berkeleyvision.org/installation.html .


* Summary / Example : Full list of commands to download and install Caffe (on Ubuntu >=14.04)
```bash
# Caffe dependencies
apt-get install libleveldb-dev libsnappy-dev libhdf5-serial-dev liblmdb-dev
apt-get install libgflags-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler
apt-get install libatlas-base-dev
apt-get install --no-install-recommends libboost-all-dev

git clone https://github.com/awabot-dev/caffe.git -b nd-pooling-public
cd caffe
mkdir build
cd build
cmake ..
make
```
