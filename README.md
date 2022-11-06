# GCV
OpenGL/OpenCL/VAAPI interop demos using the 4.x branch of OpenCV (https://github.com/opencv/opencv/tree/4.x)

The goal of the demos is to show how to use OpenCL interop in conjunction with OpenCV to create programs that run mostly (the part the matters) on the GPU. 

# Hardware requirements
* Support for cl_khr_gl_sharing and cl_intel_va_api_media_sharing OpenCL extensions.
* If you are on a recent Intel Platform (Gen8 - Gen12) you probably need to install an alternative [compute-runtime](https://github.com/kallaballa/compute-runtime)

There are currently three demos:
* tetra-demo: renders a rainbow tetrahedron on blue background using OpenGL and encodes to VP9
* video-demo: renders a rainbow tetrahedron on top of a input-video using OpenGL and encodes to VP9
* nanovg-demo: renders a color wheel on top of a input-video using nanovg (OpenGL) and encodes to VP9

# Instructions
You need to build the most recent 4.x branch of OpenCV.

## Build OpenCV

```bash
git clone --branch 4.x https://github.com/opencv/opencv.git
cd opencv
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DHAVE_EGL_INTEROP=ON -DWITH_OPENGL=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DWITH_QT=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF ..
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
