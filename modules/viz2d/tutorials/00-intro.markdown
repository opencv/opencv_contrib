# Viz2D {#viz2d}

[TOC]

# What is Viz2D?
Viz2D is a way of writing graphical (on- and offscreen) high performance applications with OpenCV. It is light-weight and unencumbered by QT or GTK licenses. It features vector graphics using [NanoVG](https://github.com/inniyah/nanovg) a GUI based on [NanoGUI](https://github.com/mitsuba-renderer/nanogui) and (on supported systems) OpenCL/OpenGL and OpenCL/VAAPI interoperability. It should be included in [OpenCV-contrib](https://github.com/opencv/opencv_contrib) once it is ready.

# Why Viz2D?
Please refer to the following online demos to see at a glance what it can do for you.

* OpenGL: Easy access to OpenGL.
* GUI: Simple yet powerful user interfaces through NanoGUI.
* Vector graphics: Elegant and fast vector graphics through NanoVG.
* Font rendering: Loading of TTF-fonts and sophisticated rendering options.
* Video pipeline: Through a simple Source/Sink system videos can be displayed, edited and saved.
* Hardware acceleration: Automatic hardware acceleration usage where possible. (e.g. cl-gl sharing and VAAPI). Actually it is possible to write programs that run almost entirely on the GPU, given driver-features are available.
* No more highgui with it's heavy dependencies, licenses and limitations.
* WebAssembly support

# Online Demos

Please note that the following online demos are slower and/or have less features than the native versions.
* https://viel-zu.org/opencv/shader
* https://viel-zu.org/opencv/font
* https://viel-zu.org/opencv/optflow
* https://viel-zu.org/opencv/beauty

# Design Notes
* Viz2D is not thread safe. Though it is possible to have several Viz2D objects in one or more threads and synchronize them using ```Viz2D::makeNonCurrent()``` and ```Viz2D::makeCurrent()```. This is a limitation of GLFW3. That said, OpenCV algorithms are multi-threaded as usual.
* Viz2D uses InputArray/OutputArray/InputOutputArray which gives you the option to work with Mat, std::vector and UMat. Anyway, you should prefer to use UMat whenever possible to automatically use hardware capabilities where available.
* Access to different subsystems (opengl, framebuffer, nanovg and nanogui) is provided through "contexts". A context is simply a function that takes a functor, sets up the subsystem, executes the functor and tears-down the subsystem.
* Contexts ***may not*** be nested.

For example, to create an OpenGL context and set the GL viewport:
@code{.cpp}
//Creates a Viz2D object for on screen rendering
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "GL viewport");

//Takes care of OpenGL states in the background
v2d->gl([](const Size sz) {
    glViewPort(0, 0, sz.width, sz.height);
});
@endcode

# Requirements
* C++20 (at the moment)
* OpenGL 4/OpenGL ES 3.0

# Optional requirements
* Support for OpenCL 1.2
* Support for cl_khr_gl_sharing and cl_intel_va_api_media_sharing OpenCL extensions.
* If you want cl-gl sharing on a recent Intel Platform (Gen8 - Gen12) you currently **need to install** [compute-runtime](https://github.com/intel/compute-runtime) from source and [my OpenCV fork](https://github.com/kallaballa/opencv)

# Dependencies
* [OpenCV 4.x](https://github.com/opencv/opencv)
* GLEW
* GLFW3
* [nanovg](https://github.com/inniyah/nanovg)
* [nanogui](https://github.com/mitsuba-renderer/nanogui)

# Examples
Those are minimal examples, full samples below.

## Display an image
Actually there are several ways to display an image but for now we focus on the most convinient way.

@include samples/cpp/display_image.cpp

This will create a window with size WIDTHxHEIGHT for on-screen rendering with the title "Show Image" and display the image (using the video pipeline which resizes the image to framebuffer size, but more about that later).

## Render OpenGL
This example renders a rotating tetrahedron using legacy OpenGL for brevity.

@include samples/cpp/render_opengl.cpp

## Manipulate the framebuffer using OpenCV/OpenCL
All contexts operate on the same framebuffer through different means. OpenCV (using OpenCL where available) can manipulate results of other contexts throught the ```fb``` context.

@include samples/cpp/manipulate_fb.cpp

## Vector graphics
Through the nvg context javascript-like canvas-rendering is possible.

@include samples/cpp/vector_graphics.cpp

## Vector graphics and framebuffer manipulation
The framebuffer can be accessed directly to manipulate data created in other contexts.

@include samples/cpp/vector_graphics_and_fb.cpp

## Font rendering
Draws "hello world" to the screen.

@include samples/cpp/font_rendering.cpp

## Video editing
Through adding a Source and a Sink v2d becomes capable of video editing. Reads a video, renders text on top and writes the result.

@include samples/cpp/video_editing.cpp

## Font rendering with form based GUI
Draws "Hello World" to the screen and let's you control the font size and color with a GUI based on FormHelper.

@include samples/cpp/font_with_gui.cpp

# Samples
The goal of the samples is to show how to use Viz2D to the fullest. Also they show how to use Viz2D in conjunction with interop options to create programs that run mostly (the part the matters) on the GPU. You ***only*** need to build my fork of OpenCV 4.x if you want to use cl-gl sharing on recent Intel platforms (Gen8 - Gen12).

There are currently eight sampes. The shader-demo, font-demo, optflow-demo and beauty-demo can be compiled to WebAssembly using Emscripten but for now you have to figure out how to do it yourself :).
