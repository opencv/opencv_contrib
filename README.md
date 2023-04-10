# Viz2D
Viz2D is a visualization module for OpenCV. It features vector graphics using [NanoVG](https://github.com/memononen/nanovg) a GUI based on [NanoGUI](https://github.com/mitsuba-renderer/nanogui) and (on supported systems) OpenCL/OpenGL and OpenCL/VAAPI interoperability, . It should be included in OpenCV-contrib once it is ready.

# What is Viz2D?
Viz2D is a way of writing graphical (on- and offscreen) high performance applications with OpenCV. It is light-weight and unencumbered by QT or GTK licenses.

# Why Viz2D?
Please refere to the [online demos](https://github.com/kallaballa/Viz2D/blob/main/README.md#online-demos) to see at a glance what it can do for you.

* OpenGL: Easy access to OpenGL.
* GUI: Simple yet powerful user interfaces through NanoGUI.
* Vector graphics: Elegant and fast vector graphics through NanoVG.
* Font rendering: Loading of TTF-fonts and sophisticated rendering options.
* Video pipeline: Through a simple Source/Sink system videos can be displayed, edited and saved.
* Hardware acceleration: Automatic hardware acceleration usage where possible. (e.g. cl-gl sharing and VAAPI). Actually it is possible to write programs to run almost entirely on the GPU, given driver-features are available.
* No more highgui with it's heavy dependencies, licenses and limitations.
* WebAssembly

# Documentation
API documentation is available [here](https://viel-zu.org/opencv/apidoc/)

## Basics
* Viz2D is not thread safe. Though it is possible to have several Viz2D objects in one or more threads and synchronize them using ```Viz2D::makeNonCurrent()``` and ```Viz2D::makeCurrent()```. This is a limitation of GLFW3. That said, OpenCV algorithms are multi-threaded as usual.
* Viz2D uses InputArray/OutputArray/InputOutputArray which gives you the option to work with Mat, std::vector and UMat. Anyway, you should prefer to use UMat whenever possible to automatically use hardware capabilities where available.
* Access to different subsystems (opengl, opencl, nanovg and nanogui) is provided through "contexts". A context is simply a function that takes a functor, sets up the subsystem, executes the functor and tears-down the subsystem.
* Contexts ***may not*** be nested.

For example, to create an OpenGL context and set the GL viewport:
```C++
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "GL viewport");

//takes care of OpenGL states in the background
v2d->gl([](const Size sz) {
    glViewPort(0, 0, sz.width, sz.height);
});
```

## Examples
Those are minimal examples, full samples below. Note that the examples for simplicity use their own run loops calling ```keepRunning()``` when in fact they should use ```Viz2D::run()``` (for portability reasons). The samples do use ```Viz2D::run()```.

### Display an images
Actually there are several ways to display an image but for now we focus on the most convinient way.

```C++
//Create a Viz2D object for on screen rendering
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Show image");
//An image
UMat image = imread("sample.png");
//Feeds the image to the video pipeline
v2d->feed(image);
//Display the framebuffer in the native window
v2d->display();
```

This will create a window with size WIDTHxHEIGHT for on-screen rendering with the title "Show Image" and display the image (using the video pipeline, but more about that later).

### Render OpenGL
This example renders a rotating tetrahedron using legacy OpenGL for brevity.

```C++
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "GL Tetrahedron");

v2d->gl([](const Size sz) {
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
    v2d->gl([](const Size sz) {
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

### Manipulate the framebuffer using OpenCV/OpenCL
All contexts operate on the same framebuffer through different means. That means that OpenCV can manipulate results of other contexts throught the ```fb``` context.

```C++
//Create a Viz2D object for on screen rendering
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Manipulate Framebuffer");
//An image
UMat image = imread("sample.png");
//Feeds the image to the video pipeline
v2d->feed(image);
//directly accesses the framebuffer using OpenCV (and using OpenCL if available)
v2d->fb([](UMat& framebuffer) {
    flip(framebuffer,framebuffer,0); //Flip the framebuffer
});
//Display the upside-down image in the native window
v2d->display();

```

### Vector graphics
Through the nvg context javascript-like canvas-rendering is possible.
```C++
//Create a Viz2D object for on screen rendering
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Vector Graphics");
//create a NanoVG context and draws a cross-hair on the framebuffer
v2d->nvg([](const Size sz) {
    //calls from this namespace may only be used inside a nvg context
    namespace viz::nvg;
    beginPath();
    strokeWidth(3.0);
    strokeColor(Scalar(0,0,255,255)); //BGRA
    moveTo(0, WIDTH/2.0);
    lineTo(HEIGHT, WIDTH/2.0);
    moveTo(HEIGHT/2.0, 0);
    lineTo(HEIGHT/2.0, WIDTH);
    stroke();
});
//display
v2d->display()
```
### Vector graphics and framebuffer manipulation
The framebuffer can be accessed directly to manipulate data created in other contexts.

```C++
//Create a Viz2D object for on screen rendering
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Vector Graphics");
//create a NanoVG context and draws a cross-hair on the framebuffer
v2d->nvg([](const Size sz) {
    //calls from this namespace may only be used inside a nvg context
    namespace viz::nvg;
    beginPath();
    strokeWidth(3.0);
    strokeColor(Scalar(0,0,255,255)); //BGRA
    moveTo(0, WIDTH/2.0);
    lineTo(HEIGHT, WIDTH/2.0);
    moveTo(HEIGHT/2.0, 0);
    lineTo(HEIGHT/2.0, WIDTH);
    stroke();
});
//directly accesses the framebuffer using OpenCV (and using OpenCL if available)
v2d->fb([](UMat& framebuffer) {
    //blurs the crosshair using a cheap boxFilter
    boxFilter(framebuffer, framebuffer, -1, Size(5, 5), Point(-1,-1), true, BORDER_REPLICATE);
});
//display
v2d->display()
```

### Font rendering
Draws "hello world" to the screen.
```C++
string hw = "hello world";
//Create a Viz2D object for on screen rendering
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Vector Graphics");
//Clear with black
v2d->clear();
//render the text at the center of the screen
v2d->nvg([&](const Size& sz) {
    using namespace viz::nvg;
    fontSize(font_size);
    fontFace("sans-bold");
    fillColor(Scalar(255, 0, 0, 255));
    textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
    text(WIDTH / 2.0, HEIGHT / 2.0, hw.c_str(), hw.c_str() + hw.size());
});
//display
v2d->display()
```

### Video editing
Through adding a Source and a Sink v2d becomes capable of video editing. Reads a video, renders text on top and writes the result.

```C++
string hw = "hello video!";
//Create a Viz2D object for on screen rendering
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Video Editing");
//Setup source and sink
//Input file
Source src = makeCaptureSource("input.webm");
//Output file
Sink sink = makeWriterSink("output.webm", VideoWriter::fourcc('V', 'P', '9', '0'), src.fps(), Size(WIDTH, HEIGHT));

//Attach source and sink
v2d->setSource(src);
v2d->setSink(sink);

while(keepRunning()) {
    if(!v2d->capture())
        break;
    v2d->nvg([&](const Size& sz) {
        using namespace viz::nvg;

        fontSize(font_size);
        fontFace("sans-bold");
        fillColor(Scalar(255, 0, 0, 255));
        textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
        text(WIDTH / 2.0, y, hw.c_str(), hw.c_str() + hw.size());
    });
    v2d->write();
    if(!v2d->display())
        break;
}
```

### Font rendering with form based GUI
Draws "hello world" to the screen and let's you control the font size and color with a GUI based on FormHelper.

```C++
//The text color. NanoGUI uses rgba with floating point
nanogui::Color textColor = {0.0, 0.0, 1.0, 1.0};
//The font size
float fontSize = 40.0f;
//The text
string hw = "hello world";
//Create a Viz2D object for on screen rendering
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Vector Graphics");
//Setup the GUI
v2d->nanogui([&](FormHelper& form) {
    //Create a light-weight dialog
    form.makeDialog(5, 30, "Settings");
    //Create a group
    form.makeGroup("Font");
    //Create a from variable. The type of widget is deduced from the variable type.
    form.makeFormVariable("Font Size", fontSize, 1.0f, 100.0f, true, "pt", "Font size of the text crawl");
    //Create a color picker
    form.makeColorPicker("Text Color", textColor, "The text color");
});

while(keepRunning()) {
    //Clear with black
    v2d->clear();
    //render the text at the center of the screen
    v2d->nvg([&](const Size& sz) {
        using namespace viz::nvg;
        fontSize(fontSize);
        fontFace("sans-bold");
        fillColor(Scalar(textColor.b() * 255, textColor.g() * 255, textColor.r() * 255, 255));
        textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
        text(WIDTH / 2.0, HEIGHT / 2.0, hw.c_str(), hw.c_str() + hw.size());
    });
    //display
    v2d->display()
}
```
# Samples
The goal of the samples is to show how to use Viz2D to the fullest. Also they show how to use Viz2D in conjunction with interop options to create programs that run mostly (the part the matters) on the GPU. You ***only*** need to build my fork of OpenCV 4.x if you want to use cl-gl sharing on recent Intel platforms (Gen8 - Gen12).

There are currently eight sampes. The shader-demo, font-demo, optflow-demo and beauty-demo can be compiled to WebAssembly using Emscripten but for now you have to figure out how to do it yourself :).

## Online Demos

Please note that the following online demos are slower and/or have less features than the native versions.
* https://viel-zu.org/opencv/shader
* https://viel-zu.org/opencv/font
* https://viel-zu.org/opencv/optflow
* https://viel-zu.org/opencv/beauty

## [tetra-demo](https://github.com/kallaballa/Viz2D/blob/main/modules/viz2d/samples/cpp/tetra/tetra-demo.cpp)
Renders a rainbow tetrahedron on blue background using OpenGL, applies a glow effect using OpenCV (OpenCL) and encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222984424-e0914bd4-72f3-4777-8a61-28dee6dd3573.mp4

## [video-demo](https://github.com/kallaballa/Viz2D/blob/main/modules/viz2d/samples/cpp/video/video-demo.cpp)
Renders a rainbow tetrahedron on top of a input-video using OpenGL, applies a glow effect using OpenCV (OpenCL) and decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222984530-c8c39997-ed3c-4675-92c9-40e4a7ea306a.mp4

## [shader-demo](https://github.com/kallaballa/Viz2D/blob/main/modules/viz2d/samples/cpp/shader/shader-demo.cpp)
Renders a mandelbrot fractal zoom. Uses shaders, OpenCL and VAAPI together.

https://user-images.githubusercontent.com/287266/222971445-13b75aee-f190-419d-9772-404d32ff61f2.mp4

## [nanovg-demo](https://github.com/kallaballa/Viz2D/blob/main/modules/viz2d/samples/cpp/nanovg/nanovg-demo.cpp)
Renders a color wheel on top of an input-video using nanovg (OpenGL), does colorspace conversions using OpenCV (OpenCL) and decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222984631-a7e3522a-8713-4413-ab5e-e6b55cd52ce8.mp4

## [font-demo](https://github.com/kallaballa/Viz2D/blob/main/modules/viz2d/samples/cpp/font/font-demo.cpp)
Renders a Star Wars like text crawl using nanovg (OpenGL), uses OpenCV (OpenCL) for a pseudo 3D effect and encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222984217-50af8dc1-72cb-4998-babe-53a2114745cf.mp4

## [optflow-demo](https://github.com/kallaballa/Viz2D/blob/main/modules/viz2d/samples/cpp/optflow/optflow-demo.cpp)
My take on a optical flow visualization on top of a video. Uses background subtraction (OpenCV/OpenCL) to isolate areas with motion, detects features to track (OpenCV/OpenCL), calculates the optical flow (OpenCV/OpenCL), uses nanovg for rendering (OpenGL) and post-processes the video (OpenCL). Decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222980868-e032bc80-0a2a-4406-b64e-7b2acdc22416.mp4

## [pedestrian-demo](https://github.com/kallaballa/Viz2D/blob/main/modules/viz2d/samples/cpp/pedestrian/pedestrian-demo.cpp)
Pedestrian detection using HOG with a linear SVM, non-maximal suppression and tracking using KCF. Uses nanovg for rendering (OpenGL), detects using a linear SVM (OpenCV/OpenCL), filters resuls using NMS (CPU) and tracks using KCF (CPU). Decodes/encodes on the GPU (VAAPI).

https://user-images.githubusercontent.com/287266/222980241-d631f7e5-e7a3-446e-937e-bce34e194bd1.mp4

## [beauty-demo](https://github.com/kallaballa/Viz2D/blob/main/modules/viz2d/samples/cpp/beauty/beauty-demo.cpp)
Face beautification using face landmark detection (OpenCV/OpenCL), nanovg (OpenGL) for drawing masks and multi-band blending (CPU) to put it all together.

https://user-images.githubusercontent.com/287266/222982914-ff5be485-4aec-4d6b-9eef-378f6b10d773.mp4

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

## Build the samples
```bash
cd Viz2D/modules/viz2d/samples/cpp
make -j
```

## Download the example file
```bash
wget -O bunny.webm https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f3/Big_Buck_Bunny_first_23_seconds_1080p.ogv/Big_Buck_Bunny_first_23_seconds_1080p.ogv.1080p.vp9.webm
```

## Run the demos

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
