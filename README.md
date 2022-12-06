# GCV
OpenGL/OpenCL/VAAPI interop demos (aka. run it on the GPU!) using my 4.x fork of OpenCV (https://github.com/kallaballa/opencv/tree/GCV)

The goal of the demos is to show how to use OpenCL interop in conjunction with OpenCV on Linux to create programs that run mostly (the part the matters) on the GPU. Until the [necessary changes](https://github.com/opencv/opencv/pulls/kallaballa) are pulled into the official repository you need to build my fork of OpenCV 4.x.

* The author of the example video (which is also used for two of the demo videos in this README) is **(c) copyright Blender Foundation | www.bigbuckbunny.org**.
* The author of the video used for pedestrian detection is **GNI Dance Company** ([Original video](https://www.youtube.com/watch?v=yg6LZtNeO_8))
* The right holders of the video used for the optical flow visualization are **https://www.bbtv.com**. I tried to contact them several times to get an opinion on my fair-use for educational purpose. ([Original video](https://www.youtube.com/watch?v=ItGwXRCcisA))

# Requirements
* Support for OpenCL 1.2
* Support for cl_khr_gl_sharing and cl_intel_va_api_media_sharing OpenCL extensions.
* If you are on a recent Intel Platform (Gen8 - Gen12) you **need to install** an [**alternative compute-runtime**](https://github.com/kallaballa/compute-runtime)

# Dependencies
tetra-demo and video-demo have no additional dependencies.
The other demos require:
* [nanogui](https://github.com/mitsuba-renderer/nanogui)

There are currently seven demos (**the preview videos are scaled down and compressed**):
## tetra-demo
Renders a rainbow tetrahedron on blue background using OpenGL, applies a glow effect using OpenCV (OpenCL) and encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/205185854-6fe42c8a-7cf7-4a35-bf13-0168a95619a2.mp4

## video-demo
Renders a rainbow tetrahedron on top of a input-video using OpenGL, applies a glow effect using OpenCV (OpenCL) and decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/205186059-c96e2728-62e4-41c5-b6a2-c7e393efeda2.mp4

## nanovg-demo
Renders a color wheel on top of an input-video using nanovg (OpenGL), does colorspace conversions using OpenCV (OpenCL) and decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/205188005-6c48d443-89b5-4ba7-81a7-252a12baa1ac.mp4

## font-demo
Renders a Star Wars like text crawl using nanovg (OpenGL), uses OpenCV (OpenCL) for a pseudo 3D effect and encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/205252994-684256b4-a0c9-4755-b4b8-ca06e56501db.mp4

## optflow-demo
My take on a optical flow visualization on top of a video. Uses background subtraction (OpenCV/OpenCL) to isolate areas with motion, detects features to track (OpenCV/OpenCL), calculates the optical flow (OpenCV/OpenCL), uses nanovg for rendering (OpenGL) and post-processes the video (OpenCL). Decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/202174513-331e6f08-8397-4521-969b-24cbc43d27fc.mp4

## pedestrian-demo
Pedestrian detection using HOG with a linear SVM and non-maximal suppression. Uses nanovg for rendering (OpenGL), detects using a linear SVM (OpenCV/OpenCL), filters resuls using NMS (CPU). Decodes/encodes on the GPU (VAAPI). 
Note: Detection rate is not very impressive and depends highly on the video.

https://user-images.githubusercontent.com/287266/204570888-9bf48c6e-3422-4fce-94e4-27a98db76dea.mp4

## beauty-demo
Face beautification using face landmark detection (OpenCV/OpenCL), nanovg (OpenGL) for drawing masks and multi-band (OpenCV/OpenCL) blending to put it all together. Note: There are sometimes little glitches because face landmark detection is not very accurate and has rather few points.

# Instructions
You need to build my 4.x branch of OpenCV.

## Build OpenCV

```bash
git clone --branch GCV https://github.com/kallaballa/opencv.git
cd opencv
mkdir build
cd build
ccmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_ENABLE_GLX=ON -DOPENCV_ENABLE_EGL=ON -DOPENCV_FFMPEG_ENABLE_LIBAVDEVICE=ON -DWITH_OPENGL=ON -DWITH_QT=ON -DWITH_FFMPEG=ON -DOPENCV_FFMPEG_SKIP_BUILD_CHECK=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF ..
make -j8
sudo make install
```

## Build demo code

```bash
git clone https://github.com/kallaballa/GCV.git
cd GCV
make
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


