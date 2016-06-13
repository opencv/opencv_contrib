Modern Deep Learning Module
===========================

The module is wrapper to [tiny-cnn ](https://github.com/nyanp/tiny-cnn)

A header only, dependency-free deep learning framework in C++11

Installation
------------
**Required Dependencies**
 - System under Unix or Windows
 - C++11 compiler
 - tiny-cnn headers

**How to install tiny-cnn?**

  Download tiny-cnn project somewhere in your system

    cd /opt
    git clone https://github.com/nyanp/tiny-cnn.git

  Run your OpenCV CMake pointing to your tiny-cnn headers location

    cd /opt/opencv/build
    cmake -DTINYCNN_ROOT=/opt/tiny-cnn ..
    make -j4

**Extra**

  You can enable some optimizations just for tiny-cnn backend

    cmake -DTINYCNN_USE_SSE=ON ..
    cmake -DTINYCNN_USE_AVX=ON ..

  Use third-party multithreading libs: TBB or OMP.

    cmake -DTINYCNN_USE_TBB=ON ..     // then disable OMP
    cmake -DTINYCNN_USE_OMP=ON ..     // then disable TBB

  NNPACK: Acceleration package for neural networks on multi-core CPUs.<br />
  Check project site for installation: [https://github.com/Maratyszcza/NNPACK](https://github.com/Maratyszcza/NNPACK)

    cmake -DTINYCNN_USE_NNPACK=ON ..  // not supported yet for Caffe loader
