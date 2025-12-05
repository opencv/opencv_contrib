# WebAssembly Support {#v4d_webassembly_support}

[TOC]

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

# What is WebAssembly?
It is possible to compile C++ (but also other languages) for the browser. The resulting binaries contain usually WebAssembly (WASM) which the browser knows to execute.

# So what makes it special for OpenCV and V4D?
For OpenCV there has been the possibility to run code in the browser for a while using [OpenCV.js](https://docs.opencv.org/4.x/d0/d84/tutorial_js_usage.html). But OpenCV.js merely offers the OpenCV APIs and visualization and GUI has to be done by other means (e.g. HTML5 Canvas). That is where V4D steps in because it has been written with WebAssembly in mind and cleverly uses [OpenGL](https://en.wikipedia.org/wiki/OpenGL) in a fashion that translates well to [WebGL](https://en.wikipedia.org/wiki/WebGL). V4D enables you to write graphical OpenCV applications that run native as well as in the browser.

# Dependencies
* [Emscripten](https://emscripten.org)
* [My OpenCV 4.x Fork](https://github.com/kallaballa/opencv/tree/GCV)
* [V4D](https://github.com/kallaballa/V4D)

# Instructions for Ubuntu 22.04.2 LTS

## Install required packages
```
# Install basic packages
apt install cmake make git-core build-essential pkg-config python3 software-properties-common
```

## Optional: Install Firefox
In case you don't have a recent browser here are the instructions to get Firefox.

```
# Add Mozilla PPA
add-apt-repository ppa:mozillateam/ppa

#Install Firefox
apt install firefox-esr
```

## Install emscripten
```
# Get the emsdk repo
git clone https://github.com/emscripten-core/emsdk.git

# Enter that directory
cd emsdk

 Download and install the latest SDK tools.
./emsdk install latest

# Make the "latest" SDK "active" for the current user. (writes .emscripten file)
./emsdk activate latest

# Activate PATH and other environment variables in the current terminal
source ./emsdk_env.sh

# Leave the directory
cd ..
```

## Build V4D with my OpenCV fork and all examples and demos for the browser
```
git clone --branch GCV https://github.com/kallaballa/opencv.git
git clone https://github.com/kallaballa/V4D.git
mkdir opencv/build
cd opencv/build
emcmake cmake -DOPENCV_FORCE_3RDPARTY_BUILD=OFF -DPYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 -DENABLE_PIC=FALSE -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_TOOLCHAIN_FILE='../../emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake' -DCPU_BASELINE='' -DCMAKE_INSTALL_PREFIX=/usr/local -DCPU_DISPATCH='' -DCV_TRACE=OFF -DBUILD_SHARED_LIBS=OFF -DWITH_OPENGL=ON -DWITH_1394=OFF -DWITH_ADE=OFF -DWITH_VTK=OFF -DWITH_EIGEN=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_GTK_2_X=OFF -DWITH_IPP=OFF -DWITH_JASPER=OFF -DWITH_JPEG=OFF -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF -DWITH_OPENVX=OFF -DWITH_OPENNI=OFF -DWITH_OPENNI2=OFF -DWITH_PNG=OFF -DWITH_TBB=OFF -DWITH_TIFF=OFF -DWITH_V4L=OFF -DWITH_OPENCL=OFF -DWITH_OPENCL_SVM=OFF -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_GPHOTO2=OFF -DWITH_LAPACK=OFF -DWITH_ITT=OFF -DWITH_QUIRC=ON -DBUILD_ZLIB=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_calib3d=ON -DBUILD_opencv_dnn=ON -DBUILD_opencv_features2d=ON -DBUILD_opencv_flann=ON -DBUILD_opencv_gapi=OFF -DBUILD_opencv_ml=OFF -DBUILD_opencv_photo=ON -DBUILD_opencv_imgcodecs=ON -DBUILD_opencv_shape=OFF -DBUILD_opencv_videoio=ON -DBUILD_opencv_videostab=OFF -DBUILD_opencv_highgui=OFF -DBUILD_opencv_superres=OFF -DBUILD_opencv_stitching=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_js=ON -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_alphamat=OFF -DBUILD_opencv_aruco=OFF -DBUILD_opencv_barcode=OFF -DBUILD_opencv_bgsegm=OFF -DBUILD_opencv_bioinspired=OFF -DBUILD_opencv_ccalib=ON -DBUILD_opencv_cnn_3dobj=OFF -DBUILD_opencv_cudaarithm=OFF -DBUILD_opencv_cudabgsegm=OFF -DBUILD_opencv_cudacodec=OFF -DBUILD_opencv_cudafeatures2d=OFF -DBUILD_opencv_cudafilters=OFF -DBUILD_opencv_cudaimgproc=OFF -DBUILD_opencv_cudalegacy=OFF -DBUILD_opencv_cudaobjdetect=OFF -DBUILD_opencv_cudaoptflow=OFF -DBUILD_opencv_cudastereo=OFF -DBUILD_opencv_cudawarping=OFF -DBUILD_opencv_cudev=OFF -DBUILD_opencv_cvv=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn_objdetect=OFF -DBUILD_opencv_dnns_easily_fooled=OFF -DBUILD_opencv_dnn_superres=OFF -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=ON -DBUILD_opencv_freetype=OFF -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_hdf=OFF -DBUILD_opencv_hfs=OFF -DBUILD_opencv_img_hash=OFF -DBUILD_opencv_intensity_transform=OFF -DBUILD_opencv_julia=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_matlab=OFF -DBUILD_opencv_mcc=OFF -DBUILD_opencv_optflow=ON -DBUILD_opencv_ovis=OFF -DBUILD_opencv_phase_unwrapping=OFF -DBUILD_opencv_plot=ON -DBUILD_opencv_quality=OFF -DBUILD_opencv_rapid=OFF -DBUILD_opencv_README.md=OFF -DBUILD_opencv_reg=OFF -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_sfm=OFF -DBUILD_opencv_shape=OFF -DBUILD_opencv_stereo=OFF -DBUILD_opencv_structured_light=OFF -DBUILD_opencv_superres=OFF -DBUILD_opencv_surface_matching=OFF -DBUILD_opencv_text=OFF -DBUILD_opencv_tracking=ON -DBUILD_opencv_videostab=OFF -DBUILD_opencv_viz=OFF -DBUILD_opencv_wechat_qrcode=OFF -DBUILD_opencv_xfeatures2d=OFF -DBUILD_opencv_ximgproc=ON -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=OFF -DBUILD_EXAMPLES=ON -DBUILD_PACKAGE=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DOPENCV_EXTRA_MODULES_PATH=../../V4D/modules/ -DBUILD_DOCS=OFF -DWITH_PTHREADS_PF=ON -DCV_ENABLE_INTRINSICS=ON -DBUILD_WASM_INTRIN_TESTS=OFF -DCMAKE_C_FLAGS="-s USE_PTHREADS=1 -s USE_ZLIB=1 -msimd128" -DCMAKE_CXX_FLAGS="-s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE_STRICT=0 -s PTHREAD_POOL_SIZE=8 -s USE_ZLIB=1 -msimd128" -DCMAKE_LD_FLAGS="-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] --bind -s MALLOC=emmalloc -s WASM_BIGINT=1 -s USE_GLFW=3 -s WASM=1 -s SINGLE_FILE=1 -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE_STRICT=0 -s PTHREAD_POOL_SIZE=8 -s USE_ZLIB=1 -msimd128" ..
make -j8
sudo make -j8 install
```

## Run the examples and demos
Though the examples and demos are compiled and come with an html file to run them, you can't just do so. Certain WebAssembly features require [special HTTP headers](https://emscripten.org/docs/porting/pthreads.html?highlight=pthreads). Anyway, emscripten provides a tool ([emrun](https://emscripten.org/docs/compiling/Running-html-files-with-emrun.html)) that serves a web server configured just the right way.

So to run cube-demo you have to do the following:

```
emrun --browser=firefox-esr bin/example_v4d_cube-demo.html
```