# Viz2D {#viz2d}

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

@code{.cpp}
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Show image");
//An image
UMat image = imread("sample.png");
//Feeds the image to the video pipeline
v2d->feed(image);
//Display the framebuffer in the native window
v2d->display();
@endcode

This will create a window with size WIDTHxHEIGHT for on-screen rendering with the title "Show Image" and display the image (using the video pipeline which resizes the image to framebuffer size, but more about that later).

### Render OpenGL
This example renders a rotating tetrahedron using legacy OpenGL for brevity.

@code{.cpp}
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
//Viz2D::run() though it takes a functor is not a context. It is simply an abstraction
//of a run loop for portability reasons and executes the functor until the application
//terminates or the functor returns false.
v2d->run([]() {
    v2d->gl([](const Size& sz) {
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
    return v2d->display();
});
@endcode

### Manipulate the framebuffer using OpenCV/OpenCL
All contexts operate on the same framebuffer through different means. OpenCV (using OpenCL where available) can manipulate results of other contexts throught the ```fb``` context.

@code{.cpp}
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Manipulate Framebuffer");
//An image
UMat image = imread("sample.png");
//Feeds the image to the video pipeline
v2d->feed(image);
//Directly access the framebuffer using OpenCV
v2d->fb([](UMat& framebuffer) {
    flip(framebuffer,framebuffer,0); //Flip the framebuffer
});
//Display the upside-down image in the native window
v2d->display();
@endcode

### Vector graphics
Through the nvg context javascript-like canvas-rendering is possible.

@code{.cpp}
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Vector Graphics");
//Creates a NanoVG context and draws a cross-hair on the framebuffer
v2d->nvg([](const Size sz) {
    //calls from this namespace may only be used inside a nvg context
    namespace cv::viz::nvg;
    beginPath();
    strokeWidth(3.0);
    strokeColor(Scalar(0,0,255,255)); //BGRA
    moveTo(0, WIDTH/2.0);
    lineTo(HEIGHT, WIDTH/2.0);
    moveTo(HEIGHT/2.0, 0);
    lineTo(HEIGHT/2.0, WIDTH);
    stroke();
});

v2d->display()
@endcode

### Vector graphics and framebuffer manipulation
The framebuffer can be accessed directly to manipulate data created in other contexts.

@code{.cpp}
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Vector Graphics");
//Creates a NanoVG context and draws a cross-hair on the framebuffer
v2d->nvg([](const Size sz) {
    namespace cv::viz::nvg;
    beginPath();
    strokeWidth(3.0);
    strokeColor(Scalar(0,0,255,255)); //BGRA
    moveTo(0, WIDTH/2.0);
    lineTo(HEIGHT, WIDTH/2.0);
    moveTo(HEIGHT/2.0, 0);
    lineTo(HEIGHT/2.0, WIDTH);
    stroke();
});
v2d->fb([](UMat& framebuffer) {
    //Blurs the crosshair using a cheap boxFilter
    boxFilter(framebuffer, framebuffer, -1, Size(5, 5), Point(-1,-1), true, BORDER_REPLICATE);
});
v2d->display()
@endcode

### Font rendering
Draws "hello world" to the screen.

@code{.cpp}
string hw = "hello world";
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Vector Graphics");
//Clear with black
v2d->clear();
//Render the text at the center of the screen
v2d->nvg([&](const Size& sz) {
    using namespace cv::viz::nvg;
    fontSize(font_size);
    fontFace("sans-bold");
    fillColor(Scalar(255, 0, 0, 255));
    textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
    text(WIDTH / 2.0, HEIGHT / 2.0, hw.c_str(), hw.c_str() + hw.size());
});
v2d->display()
@endcode

### Video editing
Through adding a Source and a Sink v2d becomes capable of video editing. Reads a video, renders text on top and writes the result.

@code{.cpp}
string hw = "hello video!";
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Video Editing");
//Make the video source
Source src = makeCaptureSource("input.webm");
//Make the video sink
Sink sink = makeWriterSink("output.webm", VideoWriter::fourcc('V', 'P', '9', '0'), src.fps(), Size(WIDTH, HEIGHT));

//Attach source and sink
v2d->setSource(src);
v2d->setSink(sink);

v2d->run([]() {
    if(!v2d->capture())
        return false;
    v2d->nvg([&](const Size& sz) {
        using namespace cv::viz::nvg;

        fontSize(font_size);
        fontFace("sans-bold");
        fillColor(Scalar(255, 0, 0, 255));
        textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
        text(WIDTH / 2.0, y, hw.c_str(), hw.c_str() + hw.size());
    });
    v2d->write();
    return v2d->display();
});
@endcode

### Font rendering with form based GUI
Draws "hello world" to the screen and let's you control the font size and color with a GUI based on FormHelper.

@code{.cpp}
Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Vector Graphics");
//The text color. NanoGUI uses rgba with floating point
nanogui::Color textColor = {0.0, 0.0, 1.0, 1.0};
//The font size
float fontSize = 40.0f;
//The text
string hw = "hello world";
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

v2d->run([]() {
    v2d->clear();
    //Render the text at the center of the screen
    v2d->nvg([&](const Size& sz) {
        using namespace cv::viz::nvg;
        fontSize(fontSize);
        fontFace("sans-bold");
        fillColor(Scalar(textColor.b() * 255, textColor.g() * 255, textColor.r() * 255, 255));
        textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
        text(WIDTH / 2.0, HEIGHT / 2.0, hw.c_str(), hw.c_str() + hw.size());
    });
    return v2d->display()
});
@endcode

# Samples
The goal of the samples is to show how to use Viz2D to the fullest. Also they show how to use Viz2D in conjunction with interop options to create programs that run mostly (the part the matters) on the GPU. You ***only*** need to build my fork of OpenCV 4.x if you want to use cl-gl sharing on recent Intel platforms (Gen8 - Gen12).

There are currently eight sampes. The shader-demo, font-demo, optflow-demo and beauty-demo can be compiled to WebAssembly using Emscripten but for now you have to figure out how to do it yourself :).
