# Viz2D
Viz2D is a 2D visualization library based on OpenCV. It features OpenCL/ OpenGL, OpenCL/VAAPI interoperability and a GUI based on nanogui. It should be included in OpenCV-contrib once it is ready. Currently work happens in [this branch](https://github.com/kallaballa/Viz2D/tree/opencv-module) where you'll find a version namespaced "cv::viz" and using Apache 2 license, that will gradually become an OpenCV module

# Attribution
* The author of the bunny video is **(c) copyright Blender Foundation | www.bigbuckbunny.org**.
* The author of the dance video is **GNI Dance Company** ([Original video](https://www.youtube.com/watch?v=yg6LZtNeO_8))
* The author of the video used in the beauty-demo video is **Kristen Leanne** ([Original video](https://www.youtube.com/watch?v=hUAT8Jm_dvw&t=11s))

# Demos
The goal of the demos is to show how to use Viz2D in conjunction with interop options on Linux to create programs that run mostly (the part the matters) on the GPU. You ***only*** need to build my fork of OpenCV 4.x if you want to use cl-gl sharing on recent Intel platforms.

There are currently eight demos. The shader-demo, font-demo, optflow-demo and beauty-demo can be compiled to WebAssembly using Emscripten but for now you have to figure out how to do it yourself :).

## Online Demos

Please note that the following online demos are slower and/or have less features than the native versions.
* https://viel-zu.org/opencv/shader
* https://viel-zu.org/opencv/font
* https://viel-zu.org/opencv/optflow
* https://viel-zu.org/opencv/beauty

# Requirements
* OpenGL 4/OpenGL ES 3.0

# Optional requirements
* Support for OpenCL 1.2
* Support for cl_khr_gl_sharing and cl_intel_va_api_media_sharing OpenCL extensions.
* If you want cl-gl sharing on a recent Intel Platform (Gen8 - Gen12) you currently **need to install** [compute-runtime](https://github.com/intel/compute-runtime) from source and [my OpenCV fork](https://github.com/kallaballa/opencv) 

# Dependencies
* [OpenCV contrib](https://github.com/opencv/opencv_contrib)
* EGL
* GLEW
* GLFW3
* [nanovg](https://github.com/inniyah/nanovg)
* [nanogui](https://github.com/mitsuba-renderer/nanogui)

## tetra-demo
Renders a rainbow tetrahedron on blue background using OpenGL, applies a glow effect using OpenCV (OpenCL) and encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222984424-e0914bd4-72f3-4777-8a61-28dee6dd3573.mp4

## video-demo
Renders a rainbow tetrahedron on top of a input-video using OpenGL, applies a glow effect using OpenCV (OpenCL) and decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222984530-c8c39997-ed3c-4675-92c9-40e4a7ea306a.mp4

## shader-demo
Renders a mandelbrot fractal zoom. Uses shaders, OpenCL and VAAPI together.

https://user-images.githubusercontent.com/287266/222971445-13b75aee-f190-419d-9772-404d32ff61f2.mp4

## nanovg-demo
Renders a color wheel on top of an input-video using nanovg (OpenGL), does colorspace conversions using OpenCV (OpenCL) and decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222984631-a7e3522a-8713-4413-ab5e-e6b55cd52ce8.mp4

## font-demo
Renders a Star Wars like text crawl using nanovg (OpenGL), uses OpenCV (OpenCL) for a pseudo 3D effect and encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222984217-50af8dc1-72cb-4998-babe-53a2114745cf.mp4

## optflow-demo
My take on a optical flow visualization on top of a video. Uses background subtraction (OpenCV/OpenCL) to isolate areas with motion, detects features to track (OpenCV/OpenCL), calculates the optical flow (OpenCV/OpenCL), uses nanovg for rendering (OpenGL) and post-processes the video (OpenCL). Decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222980868-e032bc80-0a2a-4406-b64e-7b2acdc22416.mp4

## pedestrian-demo
Pedestrian detection using HOG with a linear SVM, non-maximal suppression and tracking using KCF. Uses nanovg for rendering (OpenGL), detects using a linear SVM (OpenCV/OpenCL), filters resuls using NMS (CPU) and tracks using KCF (CPU). Decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222980241-d631f7e5-e7a3-446e-937e-bce34e194bd1.mp4

## beauty-demo
Face beautification using face landmark detection (OpenCV/OpenCL), nanovg (OpenGL) for drawing masks and multi-band blending (CPU) to put it all together.

https://user-images.githubusercontent.com/287266/222982914-ff5be485-4aec-4d6b-9eef-378f6b10d773.mp4

# Instructions for Ubuntu 22.04.2 LTS
You need to build nanovg and nanogui (optionally my 4.x fork of OpenCV with OpenCV-contrib).

## Install required packages

```bash
apt install vainfo clinfo libqt5opengl5-dev freeglut3-dev ocl-icd-opencl-dev libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libpostproc-dev libswresample-dev libswscale-dev libglfw3-dev libstb-dev libglew-dev cmake make git-core build-essential opencl-clhpp-headers pkg-config zlib1g-dev doxygen
```

## Build nanovg

```bash
git clone https://github.com/inniyah/nanovg.git
mkdir nanovg/build
cd nanovg/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
sudo make install
```

## Build nanogui

```bash
git clone --recursive https://github.com/mitsuba-renderer/nanogui.git
mkdir nanogui/build
cd nanogui/build
cmake -DCMAKE_BUILD_TYPE=Release -DNANOGUI_BACKEND=OpenGL -DNANOGUI_BUILD_EXAMPLES=OFF -DNANOGUI_BUILD_GLFW=OFF -DNANOGUI_BUILD_PYTHON=OFF ..
make -j8
sudo make install
```

## Optionally build OpenCV-fork with OpenCV-contrib

```bash
git clone --branch 4.x https://github.com/opencv/opencv_contrib.git
git clone --branch GCV https://github.com/kallaballa/opencv.git
mkdir opencv/build
cd opencv/build
cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_ENABLE_GLX=ON -DOPENCV_ENABLE_EGL=ON -DOPENCV_FFMPEG_ENABLE_LIBAVDEVICE=ON -DWITH_OPENGL=ON -DWITH_QT=ON -DWITH_FFMPEG=ON -DOPENCV_FFMPEG_SKIP_BUILD_CHECK=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ -D BUILD_opencv_aruco=OFF ..
make -j8
sudo make install
```

## Build demo code

```bash
git clone https://github.com/kallaballa/Viz2D.git
cd Viz2D
make -j8
```
## Download the example file
```bash
wget -O bunny.webm https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f3/Big_Buck_Bunny_first_23_seconds_1080p.ogv/Big_Buck_Bunny_first_23_seconds_1080p.ogv.1080p.vp9.webm
```
## Point to libviz2d.so
Before you can run the demos you have to point to ```libviz2d.so```
```bash
cd Viz2D
export LD_LIBRARY_PATH=`pwd`/src/common:$LD_LIBRARY_PATH
```

## Run the demos:

```bash
src/tetra/tetra-demo
```
```bash
src/video/video-demo bunny.webm
```
```bash
src/shader/shader-demo bunny.webm
```
```bash
src/nanovg/nanovg-demo bunny.webm
```
```bash
src/font/font-demo
```
```bash
src/optflow/optflow-demo bunny.webm
```
```bash
src/pedestrian/pedestrian-demo bunny.webm
```
```bash
src/beauty/beauty-demo bunny.webm
```
