Modern Deep Learning Module
===========================

The module is wrapper to [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn),
a header only, dependency-free deep learning framework in C++11.

Installation
------------

**Required Dependencies**
 - System under Unix or Windows
 - C++11 compiler
 - tiny-dnn headers
 - protoc compiler (part of the [protobuf](https://developers.google.com/protocol-buffers/docs/overview) library)

**How to install tiny-dnn?**
  CMake will try to download a certain version of tiny-dnn which was tested to build with OpenCV.
  If the latest version is needed, location of the tiny-dnn folder can be specified manually:

-  Download tiny-dnn project somewhere in your system
```
    cd /opt
    git clone https://github.com/tiny-dnn/tiny-dnn.git
```

-  Run your OpenCV CMake pointing to your tiny-dnn headers location
```
    cd /opt/opencv/build
    cmake -DTINYDNN_ROOT=/opt/tiny-dnn ..
    make -j4
```

**Extra**

  You can enable some optimizations just for tiny-dnn backend

    cmake -DTINYDNN_USE_SSE=ON ..
    cmake -DTINYDNN_USE_AVX=ON ..

  Use third-party multithreading libs: TBB or OMP.

    cmake -DTINYDNN_USE_TBB=ON ..     // then disable OMP
    cmake -DTINYDNN_USE_OMP=ON ..     // then disable TBB

  NNPACK: Acceleration package for neural networks on multi-core CPUs.<br />
  Check project site for installation: [https://github.com/Maratyszcza/NNPACK](https://github.com/Maratyszcza/NNPACK)

    cmake -DTINYDNN_USE_NNPACK=ON ..  // not supported yet for Caffe loader

See detailed module API documentation in http://docs.opencv.org/trunk/d1/df7/group__dnn__modern.html
