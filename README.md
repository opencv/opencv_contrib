# GCV
OpenGL/OpenCL/VAAPI interop demos (aka. run it on the GPU!) using my 4.x fork of OpenCV (https://github.com/kallaballa/opencv/tree/GCV)

The goal of the demos is to show how to use OpenCL interop in conjunction with OpenCV on Linux to create programs that run mostly (the part the matters) on the GPU until there [necessary changes](https://github.com/opencv/opencv/pulls/kallaballa) are pulled into the official repository.

The author of the example video (which is also used for the demos videos in this README) is **(c) copyright Blender Foundation | www.bigbuckbunny.org**.

# Requirements
* Support for OpenCL 1.2
* Support for cl_khr_gl_sharing and cl_intel_va_api_media_sharing OpenCL extensions.
* If you are on a recent Intel Platform (Gen8 - Gen12) you need to install an alternative [compute-runtime](https://github.com/kallaballa/compute-runtime)

There are currently four demos (**the preview videos are scaled down and highly compressed**):
## tetra-demo
Renders a rainbow tetrahedron on blue background using OpenGL and decodes/encodes on the GPU.

https://user-images.githubusercontent.com/287266/200169105-2bb88288-cb07-49bb-97ef-57ac61a0cfb8.mp4

## video-demo
Renders a rainbow tetrahedron on top of a input-video using OpenGL and decodes/encodes on the GPU.

https://user-images.githubusercontent.com/287266/200169164-231cb4d8-db5c-444b-8aff-55c9f1a822cc.mp4

## nanovg-demo
Renders a color wheel on top of a input-video using nanovg (OpenGL) and decodes/encodes on the GPU.

https://user-images.githubusercontent.com/287266/200169216-1ff25db5-f5e0-49d1-92ba-ab7903168754.mp4

## optflow-demo
My take on a optical flow visualization on top of a video. Uses nanovg for rendering (OpenGL) and decodes/encodes on the GPU.

https://user-images.githubusercontent.com/287266/200934845-2cd16ad3-3d92-4529-90aa-ecd48670800f.mp4

# Instructions
You need to build the most recent 4.x branch of OpenCV.

## Build OpenCV

```bash
git clone --branch GCV https://github.com/kallaballa/opencv.git
cd opencv
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_ENABLE_EGL_INTEROP=ON -DOPENCV_FFMPEG_ENABLE_LIBAVDEVICE=ON -DWITH_OPENGL=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DWITH_QT=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF ..
make -j8
sudo make install
```

## Build demo code

```bash
git clone https://github.com/kallaballa/GCV.git
cd GCV
make -j2
```
## Download the example file
```bash
wget -O bunny.webm https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f3/Big_Buck_Bunny_first_23_seconds_1080p.ogv/Big_Buck_Bunny_first_23_seconds_1080p.ogv.1080p.vp9.webm
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

## Run the optflow-demo:

```bash
src/optflow/optflow-demo bunny.webm
```


