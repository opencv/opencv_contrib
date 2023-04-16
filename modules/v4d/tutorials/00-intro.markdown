# V4D {#v4d}

[TOC]

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

# What is V4D?
V4D offers a way of writing graphical (on- and offscreen) high performance applications with OpenCV. It is light-weight and unencumbered by QT or GTK licenses. It features vector graphics using [NanoVG](https://github.com/inniyah/nanovg) a GUI based on [NanoGUI](https://github.com/mitsuba-renderer/nanogui) and (on supported systems) OpenCL/OpenGL and OpenCL/VAAPI interoperability. It should be included in [OpenCV-contrib](https://github.com/opencv/opencv_contrib) once it is ready.

# Showcase
Please note that all renderings and videos were created on an Intel Tigerlake CPU and an Intel Iris Xe iGPU. Also the demos in the videos might run slower for various reasons (better implementation by now, screen capturing, etc.) than they would normally do.

@youtube{yYnWkkZSK7Q}

# Why V4D?
Please refer to the online demos in the following section to see at a glance what it can do for you.

* **OpenGL**: Easy access to OpenGL.
* **GUI**: Simple yet powerful user interfaces through NanoGUI.
* **Vector graphics**: Elegant and fast vector graphics through NanoVG.
* **Font rendering**: Loading of fonts and sophisticated rendering options.
* **Video pipeline**: Through a simple Source/Sink system videos can be efficently displayed, edited and saved.
* **Hardware acceleration**: Transparent hardware acceleration usage where possible. (e.g. CL-GL sharing and VAAPI). Actually it is possible to write programs that run almost entirely on the GPU, given driver-features are available.
* **No more highgui** with it's heavy dependencies, licenses and limitations.
* **WebAssembly support**.

# Online Demos

Please note that the following online demos are slower and/or have less features than the native versions.
* https://viel-zu.org/opencv/shader
* https://viel-zu.org/opencv/font
* https://viel-zu.org/opencv/optflow
* https://viel-zu.org/opencv/beauty

# Design Notes
* V4D is not thread safe. Though it is possible to have several V4D objects in one or more threads and synchronize them using ```V4D::makeNonCurrent()``` and ```V4D::makeCurrent()```. This is a limitation of GLFW3. That said, OpenCV algorithms are multi-threaded as usual.
* V4D uses InputArray/OutputArray/InputOutputArray which gives you the option to work with Mat, std::vector and UMat. Anyway, you should prefer to use UMat whenever possible to automatically use hardware capabilities where available.
* Access to different subsystems (opengl, framebuffer, nanovg and nanogui) is provided through "contexts". A context is simply a function that takes a functor, sets up the subsystem, executes the functor and tears-down the subsystem.
* ```V4D::run``` is not a context. It is an abstraction of a run loop that takes a functor and runs until the application terminates or the functor returns false. This is necessary for portability reasons.
* Contexts ***may not*** be nested.

For example, to create an OpenGL context and set the GL viewport:
@code{.cpp}
//Creates a V4D object for on screen rendering
Ptr<V4D> v4d = V4D::make(Size(WIDTH, HEIGHT), "GL viewport");

//Takes care of OpenGL states in the background
v4d->gl([](const Size sz) {
    glViewPort(0, 0, sz.width, sz.height);
});
@endcode

# Requirements
* C++20 (at the moment)
* OpenGL 3.2 Core (optionally Compat)/OpenGL ES 3.0 Core

# Optional requirements
* Support for OpenCL 1.2
* Support for cl_khr_gl_sharing and cl_intel_va_api_media_sharing OpenCL extensions.


# Dependencies
* [OpenCV 4.x](https://github.com/opencv/opencv)
* GLEW
* GLFW3
* [NanoGUI](https://github.com/mitsuba-renderer/nanogui) (includes NanoVG)

# Optional: Dependencies for samples
* [OpenCV Contrib 4.x](https://github.com/opencv/opencv_contrib)
* If you want CL-GL sharing on a recent Intel Platform (Gen8 - Gen12) you currently **need to build** [compute-runtime](https://github.com/intel/compute-runtime) and [my OpenCV 4.x fork](https://github.com/kallaballa/opencv/tree/GCV)

# Tutorials
The tutorials are designed to be read one after the other to give you a good overview over the key concepts of V4D. After that you can move on to the samples.

* \ref v4d_display_image
* \ref v4d_vector_graphics
* \ref v4d_render_opengl
* \ref v4d_font_rendering
* \ref v4d_video_editing
* \ref v4d_custom_source_and_sink
* \ref v4d_font_with_gui

# Samples
The goal of the samples is to show how to use V4D to the fullest. Also they show how to use V4D to create programs that run mostly (the part the matters) on the GPU (when driver capabilities allow). They are also a good starting point for your own applications because they touch many key aspects and algorithms of OpenCV.

* \ref v4d_tetra
* \ref v4d_video
* \ref v4d_nanovg
* \ref v4d_shader
* \ref v4d_font
* \ref v4d_pedestrian
* \ref v4d_optflow
* \ref v4d_beauty


# Instructions for Ubuntu 22.04.2 LTS
You need to build OpenCV with V4D

## Install required packages

```bash
apt install vainfo clinfo libqt5opengl5-dev freeglut3-dev ocl-icd-opencl-dev libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libpostproc-dev libswresample-dev libswscale-dev libglfw3-dev libstb-dev libglew-dev cmake make git-core build-essential opencl-clhpp-headers pkg-config zlib1g-dev doxygen libxinerama-dev libxcursor-dev libxi-dev libva-dev
```
## Build OpenCV with V4D

```bash
git clone --branch 4.x https://github.com/opencv/opencv.git
git clone https://github.com/kallaballa/V4D.git
mkdir opencv/build
cd opencv/build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_opencv_v4d=ON -DBUILD_opencv_python_tests=OFF -DBUILD_opencv_js_bindings_generator=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python3=OFF -DOPENCV_ENABLE_GLX=ON -DOPENCV_FFMPEG_ENABLE_LIBAVDEVICE=ON -DWITH_OPENGL=ON -DWITH_QT=ON -DWITH_FFMPEG=ON -DOPENCV_FFMPEG_SKIP_BUILD_CHECK=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DOPENCV_EXTRA_MODULES_PATH=../../V4D/modules/ ..
make -j8
sudo make install
```

## Optional: Make the samples

```bash
cd V4D/modules/v4d/samples/cpp
make -j
```

### Download the example file
```bash
wget -O bunny.webm https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f3/Big_Buck_Bunny_first_23_seconds_1080p.ogv/Big_Buck_Bunny_first_23_seconds_1080p.ogv.1080p.vp9.webm
```

### Run the samples

```bash
modules/v4d/samples/cpp/tetra/tetra-demo
```
```bash
modules/v4d/samples/cpp/video/video-demo bunny.webm
```
```bash
modules/v4d/samples/cpp/shader/shader-demo bunny.webm
```
```bash
modules/v4d/samples/cpp/nanovg/nanovg-demo bunny.webm
```
```bash
modules/v4d/samples/cpp/font/font-demo
```
```bash
modules/v4d/samples/cpp/optflow/optflow-demo bunny.webm
```
```bash
modules/v4d/samples/cpp/pedestrian/pedestrian-demo bunny.webm
```
```bash
modules/v4d/samples/cpp/beauty/beauty-demo bunny.webm
```

# Attribution
* The author of the bunny video is the **Blender Foundation** ([Original video](https://www.bigbuckbunny.org)).
* The author of the dance video is **GNI Dance Company** ([Original video](https://www.youtube.com/watch?v=yg6LZtNeO_8)).
* The author of the video used in the beauty-demo video is **Kristen Leanne** ([Original video](https://www.youtube.com/watch?v=hUAT8Jm_dvw&t=11s)).
* The author of cxxpool is **Copyright (c) 2022 Christian Blume**: ([LICENSE](https://github.com/bloomen/cxxpool/blob/master/LICENSE))
