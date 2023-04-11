# Viz2D
Viz2D is a visualization module for [OpenCV](https://github.com/opencv/opencv). It features vector graphics using [NanoVG](https://github.com/memononen/nanovg) a GUI based on [NanoGUI](https://github.com/mitsuba-renderer/nanogui) and (on supported systems) OpenCL/OpenGL and OpenCL/VAAPI interoperability. It should be included in [OpenCV-contrib](https://github.com/opencv/opencv_contrib) once it is ready.

# What is Viz2D?
Viz2D is a way of writing graphical (on- and offscreen) high performance applications with OpenCV. It is light-weight and unencumbered by QT or GTK licenses.

# Why Viz2D?
Please refer to the [online demos](https://github.com/kallaballa/Viz2D/blob/main/README.md#online-demos) to see at a glance what it can do for you.

* OpenGL: Easy access to OpenGL.
* GUI: Simple yet powerful user interfaces through NanoGUI.
* Vector graphics: Elegant and fast vector graphics through NanoVG.
* Font rendering: Loading of TTF-fonts and sophisticated rendering options.
* Video pipeline: Through a simple Source/Sink system videos can be displayed, edited and saved.
* Hardware acceleration: Automatic hardware acceleration usage where possible. (e.g. cl-gl sharing and VAAPI). Actually it is possible to write programs that run almost entirely on the GPU, given driver-features are available.
* No more highgui with it's heavy dependencies, licenses and limitations.
* WebAssembly

# Documentation
OpenCV module documentation with tutorials and samples is available [here](https://viel-zu.org/opencv/apidoc/)

## Online Demos

Please note that the following online demos are slower and/or have less features than the native versions.
* https://viel-zu.org/opencv/shader
* https://viel-zu.org/opencv/font
* https://viel-zu.org/opencv/optflow
* https://viel-zu.org/opencv/beauty

# Build

## Requirements
* C++20 (at the moment)
* OpenGL 4/OpenGL ES 3.0

## Optional requirements
* Support for OpenCL 1.2
* Support for cl_khr_gl_sharing and cl_intel_va_api_media_sharing OpenCL extensions.
* If you want cl-gl sharing on a recent Intel Platform (Gen8 - Gen12) you currently **need to install** [compute-runtime](https://github.com/intel/compute-runtime) from source and [my OpenCV fork](https://github.com/kallaballa/opencv) 

## Dependencies
* [OpenCV 4.x](https://github.com/opencv/opencv)
* EGL
* GLEW
* GLFW3
* [nanovg](https://github.com/inniyah/nanovg)
* [nanogui](https://github.com/mitsuba-renderer/nanogui)

## Instructions for Ubuntu 22.04.2 LTS
You need to build nanovg, nanogui and OpenCV with Viz2D

### Install required packages

```bash
apt install vainfo clinfo libqt5opengl5-dev freeglut3-dev ocl-icd-opencl-dev libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libpostproc-dev libswresample-dev libswscale-dev libglfw3-dev libstb-dev libglew-dev cmake make git-core build-essential opencl-clhpp-headers pkg-config zlib1g-dev doxygen
```

### Build nanovg

```bash
git clone https://github.com/inniyah/nanovg.git
mkdir nanovg/build
cd nanovg/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
sudo make install
```

### Build nanogui

```bash
git clone --recursive https://github.com/mitsuba-renderer/nanogui.git
mkdir nanogui/build
cd nanogui/build
cmake -DCMAKE_BUILD_TYPE=Release -DNANOGUI_BACKEND=OpenGL -DNANOGUI_BUILD_EXAMPLES=OFF -DNANOGUI_BUILD_GLFW=OFF -DNANOGUI_BUILD_PYTHON=OFF ..
make -j8
sudo make install
```

### Build OpenCV with Viz2D using C++20

```bash
git clone --branch 4.x https://github.com/opencv/opencv.git
git clone https://github.com/kallaballa/Viz2D.git
mkdir opencv/build
cd opencv/build
cmake -DCMAKE_CXX_STANDARD=20 -DCMAKE_BUILD_TYPE=Release -DBUILD_opencv_viz2d=ON -DBUILD_opencv_python_tests=OFF -DBUILD_opencv_js_bindings_generator=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python3=OFF -DOPENCV_ENABLE_GLX=ON -DOPENCV_FFMPEG_ENABLE_LIBAVDEVICE=ON -DWITH_OPENGL=ON -DWITH_QT=ON -DWITH_FFMPEG=ON -DOPENCV_FFMPEG_SKIP_BUILD_CHECK=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DOPENCV_EXTRA_MODULES_PATH=../../Viz2D/modules/ ..
make -j8
sudo make install
```

### Build the samples
```bash
cd Viz2D/modules/viz2d/samples/cpp
make -j
```

### Download the example file
```bash
wget -O bunny.webm https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f3/Big_Buck_Bunny_first_23_seconds_1080p.ogv/Big_Buck_Bunny_first_23_seconds_1080p.ogv.1080p.vp9.webm
```

### Run the demos

```bash
Viz2D/modules/viz2d/samples/cpp/tetra/tetra-demo
```
```bash
Viz2D/modules/viz2d/samples/cpp/video/video-demo bunny.webm
```
```bash
Viz2D/modules/viz2d/samples/cpp/shader/shader-demo bunny.webm
```
```bash
Viz2D/modules/viz2d/samples/cpp/nanovg/nanovg-demo bunny.webm
```
```bash
Viz2D/modules/viz2d/samples/cpp/font/font-demo
```
```bash
Viz2D/modules/viz2d/samples/cpp/optflow/optflow-demo bunny.webm
```
```bash
Viz2D/modules/viz2d/samples/cpp/pedestrian/pedestrian-demo bunny.webm
```
```bash
Viz2D/modules/viz2d/samples/cpp/beauty/beauty-demo bunny.webm
```

# Attribution
* The author of the bunny video is **(c) copyright Blender Foundation | www.bigbuckbunny.org**.
* The author of the dance video is **GNI Dance Company** ([Original video](https://www.youtube.com/watch?v=yg6LZtNeO_8))
* The author of the video used in the beauty-demo video is **Kristen Leanne** ([Original video](https://www.youtube.com/watch?v=hUAT8Jm_dvw&t=11s))
