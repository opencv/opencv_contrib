# Viz2D {#viz2d}

[TOC]

# What is Viz2D?
Viz2D offers a way of writing graphical (on- and offscreen) high performance applications with OpenCV. It is light-weight and unencumbered by QT or GTK licenses. It features vector graphics using [NanoVG](https://github.com/inniyah/nanovg) a GUI based on [NanoGUI](https://github.com/mitsuba-renderer/nanogui) and (on supported systems) OpenCL/OpenGL and OpenCL/VAAPI interoperability. It should be included in [OpenCV-contrib](https://github.com/opencv/opencv_contrib) once it is ready.

# Why Viz2D?
Please refer to the following online demos to see at a glance what it can do for you.

* OpenGL: Easy access to OpenGL.
* GUI: Simple yet powerful user interfaces through NanoGUI.
* Vector graphics: Elegant and fast vector graphics through NanoVG.
* Font rendering: Loading of TTF-fonts and sophisticated rendering options.
* Video pipeline: Through a simple Source/Sink system videos can be displayed, edited and saved.
* Hardware acceleration: Automatic hardware acceleration usage where possible. (e.g. cl-gl sharing and VAAPI). Actually it is possible to write programs that run almost entirely on the GPU, given driver-features are available.
* No more highgui with it's heavy dependencies, licenses and limitations.
* WebAssembly support.

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

# Tutorials

* \ref viz2d_display_image
* \ref viz2d_vector_graphics
* \ref viz2d_render_opengl
* \ref viz2d_font_rendering
* \ref viz2d_video_editing
* \ref viz2d_font_with_gui

# Samples
The goal of the samples is to show how to use Viz2D to the fullest. Also they show how to use Viz2D in conjunction with interop options to create programs that run mostly (the part the matters) on the GPU.

* \ref viz2d_tetra
* \ref viz2d_video
* \ref viz2d_nanovg
* \ref viz2d_shader
* \ref viz2d_font
* \ref viz2d_pedestrian
* \ref viz2d_optflow
* \ref viz2d_beauty

# Attribution
* The author of the bunny video is **(c) copyright Blender Foundation | www.bigbuckbunny.org**.
* The author of the dance video is **GNI Dance Company** ([Original video](https://www.youtube.com/watch?v=yg6LZtNeO_8))
* The author of the video used in the beauty-demo video is **Kristen Leanne** ([Original video](https://www.youtube.com/watch?v=hUAT8Jm_dvw&t=11s))
