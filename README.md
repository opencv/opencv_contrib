# Viz2D
Viz2D is a visualization module for OpenCV. It features OpenCL/OpenGL, OpenCL/VAAPI interoperability, vector graphics using NanoVG and a GUI based on NanoGUI. It should be included in OpenCV-contrib once it is ready.

# What is Viz2D?
Viz2D is a new way of writing graphical (on- and offscreen) applications with OpenCV. It is light-weight and unencumbered by problematic licenses.

# Why Viz2D?
* OpenGL: Easy access to OpenGL
* GUI: Simple yet powerful user interfaces through NanoGUI
* Vector graphics: Elegant and fast vector graphics through NanoVG
* Video pipeline: Through a simple Source/Sink system videos can be displayed, edited and saved.
* Hardware acceleration: Automatic hardware acceleration usage where possible. (e.g. cl-gl sharing and VAAPI). Actually it is possible to write programs to run almost entirely on the GPU, given driver-features are available.
* No more highgui with it's heavy dependencies, licenses and limitations.

# Documentation

## Basics
* Viz2D is not thread safe. Though it is possible to have several Viz2D objects in one or more threads and synchronize them using ```Viz2D::makeNonCurrent()``` and ```Viz2D::makeCurrent()```. This is a limitation of GLFW3. That said, OpenCV algorithms are multi-threaded as usual.
* Viz2D uses InputArray/OutputArray/InputOutputArray which gives you the option to work with cv::Mat, std::vector and cv::UMat. Anyway, you should prefer to use cv::UMat whenever possible to automatically use hardware capabilities where available.
* Access to different subsystems (opengl, opencl, nanovg and nanogui) is provided through "contexts". A context is simply a function that takes a functor, sets up the subsystem, executes the functor and tears-down the subsystem.

For example, to create an OpenGL context and set the GL viewport:
```C++
Ptr<Viz2D> v2d = Viz2D::make(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), false, "GL viewport");
v2d->setVisible(true);
//takes care of OpenGL states in the background
v2d->gl([](const cv::Size sz) {
    glViewPort(0, 0, sz.width, sz.height);
});
```

## Examples

### Display an images
Actually there are several ways to display an image but for now we focus on the most convinient way.

```C++
//An image
cv::UMat image;
//Create a Viz2D object for on screen rendering
Ptr<Viz2D> v2d = Viz2D::make(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), false, "Show image");
v2d->setVisible(true);
//Feeds the image to the video pipeline
v2d->feed(image);
//Display the framebuffer in the native window
v2d->display();
```

This will create a window with size WIDTHxHEIGHT for on-screen rendering with the title "Show Image" and display the image (using the video pipeline, but more about that later).

### Render OpenGL

This demo renders a rotating tetrahedron using legacy OpenGL for brevity.
```C++
Ptr<Viz2D> v2d = Viz2D::make(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), false, "GL viewport");
v2d->setVisible(true);

v2d->gl([](const cv::Size sz) {
    //Initialize the OpenGL scene
    glViewport(0, 0, sz.width, sz.height);
    glColor3f(1.0, 1.0, 1.0);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-2, 2, -1.5, 1.5, 1, 40);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, 0, -3);
    glRotatef(50, 1, 0, 0);
    glRotatef(70, 0, 1, 0);
});

while (keepRunning()) {
    v2d->gl([](const cv::Size sz) {
        //Render a tetrahedron using immediate mode because the code is more concise
        glViewport(0, 0, sz.width, sz.height);
        glRotatef(1, 0, 1, 0);
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBegin(GL_TRIANGLE_STRIP);
            glColor3f(1, 1, 1);
            glVertex3f(0, 2, 0);
            glColor3f(1, 0, 0);
            glVertex3f(-1, 0, 1);
            glColor3f(0, 1, 0);
            glVertex3f(1, 0, 1);
            glColor3f(0, 0, 1);
            glVertex3f(0, 0, -1.4);
            glColor3f(1, 1, 1);
            glVertex3f(0, 2, 0);
            glColor3f(1, 0, 0);
            glVertex3f(-1, 0, 1);
        glEnd();
    });

    //If onscreen rendering is enabled it displays the framebuffer in the native window.
    //Returns false if the window was closed.
    if (!v2d->display())
        break;
}
```


# Attribution
* The author of the bunny video is **(c) copyright Blender Foundation | www.bigbuckbunny.org**.
* The author of the dance video is **GNI Dance Company** ([Original video](https://www.youtube.com/watch?v=yg6LZtNeO_8))
* The author of the video used in the beauty-demo video is **Kristen Leanne** ([Original video](https://www.youtube.com/watch?v=hUAT8Jm_dvw&t=11s))

# Demos
The goal of the demos is to show how to use Viz2D in conjunction with interop options on Linux to create programs that run mostly (the part the matters) on the GPU. You ***only*** need to build my fork of OpenCV 4.x if you want to use cl-gl sharing on recent Intel platforms (Gen8 - Gen12).

There are currently eight demos. The shader-demo, font-demo, optflow-demo and beauty-demo can be compiled to WebAssembly using Emscripten but for now you have to figure out how to do it yourself :).

## Online Demos

Please note that the following online demos are slower and/or have less features than the native versions.
* https://viel-zu.org/opencv/shader
* https://viel-zu.org/opencv/font
* https://viel-zu.org/opencv/optflow
* https://viel-zu.org/opencv/beauty

# Requirements
* C++20 (at the moment)
* OpenGL 4/OpenGL ES 3.0

# Optional requirements
* Support for OpenCL 1.2
* Support for cl_khr_gl_sharing and cl_intel_va_api_media_sharing OpenCL extensions.
* If you want cl-gl sharing on a recent Intel Platform (Gen8 - Gen12) you currently **need to install** [compute-runtime](https://github.com/intel/compute-runtime) from source and [my OpenCV fork](https://github.com/kallaballa/opencv) 

# Dependencies
* [OpenCV 4.x](https://github.com/opencv/opencv)
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
You need to build nanovg, nanogui and OpenCV with Viz2D

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

## Build OpenCV with Viz2D using C++20

```bash
git clone --branch 4.x https://github.com/opencv/opencv.git
git clone https://github.com/kallaballa/Viz2D.git
mkdir opencv/build
cd opencv/build
cmake -DCMAKE_CXX_STANDARD=20 -DCMAKE_BUILD_TYPE=Release -DBUILD_opencv_viz2d=ON -DBUILD_opencv_python_tests=OFF -DBUILD_opencv_js_bindings_generator=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python3=OFF -DOPENCV_ENABLE_GLX=ON -DOPENCV_FFMPEG_ENABLE_LIBAVDEVICE=ON -DWITH_OPENGL=ON -DWITH_QT=ON -DWITH_FFMPEG=ON -DOPENCV_FFMPEG_SKIP_BUILD_CHECK=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DOPENCV_EXTRA_MODULES_PATH=../../Viz2D/modules/ ..
make -j8
sudo make install
```

## Download the example file
```bash
wget -O bunny.webm https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f3/Big_Buck_Bunny_first_23_seconds_1080p.ogv/Big_Buck_Bunny_first_23_seconds_1080p.ogv.1080p.vp9.webm
```

## Run the demos

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
