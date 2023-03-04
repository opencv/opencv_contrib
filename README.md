# GCV
OpenGL/OpenCL/VAAPI interop demos (aka. run it on the GPU!) using my 4.x fork of OpenCV (https://github.com/kallaballa/opencv/tree/GCV)

The goal of the demos is to show how to use OpenCL interop in conjunction with OpenCV on Linux to create programs that run mostly (the part the matters) on the GPU. Until the [necessary changes](https://github.com/opencv/opencv/pulls/kallaballa) are pulled into the official repository you need to build my fork of OpenCV 4.x.

* The author of the bunny video is **(c) copyright Blender Foundation | www.bigbuckbunny.org**.
* The author of the dance video is **GNI Dance Company** ([Original video](https://www.youtube.com/watch?v=yg6LZtNeO_8))

# Requirements
* Support for OpenCL 1.2
* Support for cl_khr_gl_sharing and cl_intel_va_api_media_sharing OpenCL extensions.
* If you are on a recent Intel Platform (Gen8 - Gen12) you **need to install** an [**alternative compute-runtime**](https://github.com/kallaballa/compute-runtime)

# Dependencies
* EGL
* GLEW
* GLFW3
* [nanovg](https://github.com/inniyah/nanovg)
* [nanogui](https://github.com/mitsuba-renderer/nanogui)

There are currently eight demos.

## tetra-demo
Renders a rainbow tetrahedron on blue background using OpenGL, applies a glow effect using OpenCV (OpenCL) and encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/208234618-49284df0-5fc7-4381-8c9a-063df7400a88.mp4

## video-demo
Renders a rainbow tetrahedron on top of a input-video using OpenGL, applies a glow effect using OpenCV (OpenCL) and decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/208234606-ce0babf0-69ae-4482-b7f6-3532d5464248.mp4

## shader-demo
Renders a mandelbrot fractal zoom. Uses shaders, OpenCL and VAAPI together.

https://user-images.githubusercontent.com/287266/222869740-07461be3-0477-4396-b83b-b57d1382ec7d.mp4

## nanovg-demo
Renders a color wheel on top of an input-video using nanovg (OpenGL), does colorspace conversions using OpenCV (OpenCL) and decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222870115-389f9561-79f6-4a4f-a798-dfc0e3680369.mp4

## font-demo
Renders a Star Wars like text crawl using nanovg (OpenGL), uses OpenCV (OpenCL) for a pseudo 3D effect and encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/208361370-229b46f6-83c5-4587-9687-67c14813766e.mp4

## optflow-demo
My take on a optical flow visualization on top of a video. Uses background subtraction (OpenCV/OpenCL) to isolate areas with motion, detects features to track (OpenCV/OpenCL), calculates the optical flow (OpenCV/OpenCL), uses nanovg for rendering (OpenGL) and post-processes the video (OpenCL). Decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/208234553-3669df17-dbea-4166-aaf1-e2d5c447e9f0.mp4

## pedestrian-demo
Pedestrian detection using HOG with a linear SVM and non-maximal suppression. Uses nanovg for rendering (OpenGL), detects using a linear SVM (OpenCV/OpenCL), filters resuls using NMS (CPU). Decodes/encodes on the GPU (VAAPI).
Note: Detection rate is not very impressive and depends highly on the video.

https://user-images.githubusercontent.com/287266/208234590-f76bc0ef-f356-4d8d-a280-aab57a2fbae3.mp4

## beauty-demo
Face beautification using face landmark detection (OpenCV/OpenCL), nanovg (OpenGL) for drawing masks and multi-band (OpenCV/OpenCL) blending to put it all together. Note: There are sometimes little glitches because face landmark detection is not very accurate and has rather few points.

# Instructions
You need to build my 4.x branch of OpenCV, OpenCV-contrib, nanovg and nanogui.

## Install build dependencies

```bash
apt install ubuntu-restricted-extras libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libpostproc-dev libswresample-dev libswscale-dev libglfw3-dev libstb-dev libglew-dev cmake make git-core build-essential opencl-clhpp-headers pkg-config zlib1g-dev doxygen
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

## Build OpenCV-fork with OpenCV-contrib

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
git clone https://github.com/kallaballa/GCV.git
cd GCV
make -j8
```
## Download the example file
```bash
wget -O bunny.webm https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f3/Big_Buck_Bunny_first_23_seconds_1080p.ogv/Big_Buck_Bunny_first_23_seconds_1080p.ogv.1080p.vp9.webm
```
## Run the shader-demo:

```bash
src/shader/shader-demo
```

## Run the tetra-demo:

```bash
src/tetra/tetra-demo
```

## Run the video-demo:

```bash
src/video/video-demo bunny.webm
```

## Run the nanovg-demo:

```bash
src/nanovg/nanovg-demo bunny.webm
```

## Run the font-demo:

```bash
src/font/font-demo
```

## Run the optflow-demo:

```bash
src/optflow/optflow-demo bunny.webm
```

## Run the pedestrian-demo:

```bash
src/pedestrian/pedestrian-demo bunny.webm
```

## Run the beauty-demo:

```bash
src/beauty/beauty-demo bunny.webm
```

