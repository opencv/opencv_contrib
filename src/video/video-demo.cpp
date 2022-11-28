#define CL_TARGET_OPENCL_VERSION 120

#include "../common/subsystems.hpp"
#include <string>

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr const int VA_HW_DEVICE_INDEX = 0;
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "video-demo.mkv";
constexpr unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

constexpr int GLOW_KERNEL_SIZE = std::max(int(DIAG / 138 % 2 == 0 ? DIAG / 138 + 1 : DIAG / 138), 1);

using std::cerr;
using std::endl;
using std::string;

void init_scene(unsigned long w, unsigned long h) {
    glViewport(0, 0, w, h);
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
}

void render_scene(unsigned long w, unsigned long h) {
    //Render a tetrahedron using immediate mode because the code is more concise for a demo
    glViewport(0, 0, w, h);
    glRotatef(1, 0, 1, 0);
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);

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
}

void glow_effect(const cv::UMat &src, cv::UMat &dst, const int ksize) {
    static cv::UMat resize;
    static cv::UMat blur;
    static cv::UMat dst16;

    cv::bitwise_not(src, dst);

    //Resize for some extra performance
    cv::resize(dst, resize, cv::Size(), 0.5, 0.5);
    //Cheap blur
    cv::boxFilter(resize, resize, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    //Back to original size
    cv::resize(resize, blur, src.size());

    //Multiply the src image with a blurred version of itself
    cv::multiply(dst, blur, dst16, 1, CV_16U);
    //Normalize and convert back to CV_8U
    cv::divide(dst16, cv::Scalar::all(255.0), dst, 1, CV_8U);

    cv::bitwise_not(dst, dst);
}

int main(int argc, char **argv) {
    using namespace kb;

    if(argc != 2) {
        cerr << "Usage: video-demo <video-file>" << endl;
        exit(1);
    }

    kb::init(WIDTH, HEIGHT);

    //Initialize MJPEG HW decoding using VAAPI
    cv::VideoCapture capture(argv[1], cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    //Copy OpenCL Context for VAAPI. Must be called right after first VideoWriter/VideoCapture initialization.
    va::copy();

    if (!capture.isOpened()) {
        cerr << "ERROR! Unable to open video input" << endl;
        return -1;
    }

    double fps = capture.get(cv::CAP_PROP_FPS);

    //Initialize VP9 HW encoding using VAAPI. We don't need to specify the hardware device twice. only generates a warning.
    cv::VideoWriter writer(OUTPUT_FILENAME, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, cv::Size(WIDTH, HEIGHT), {
            cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    //If we are rendering offscreen we don't need x11
    if(!OFFSCREEN)
        x11::init("video-demo");

    //Passing true to init_egl will create a OpenGL debug context
    egl::init();
    //Initialize OpenCL Context for OpenGL
    gl::init();

    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    init_scene(WIDTH, HEIGHT);

    cv::UMat frameBuffer;
    cv::UMat videoFrame;
    cv::UMat videoFrameRGBA;

    //Activate the OpenCL context for VAAPI
    va::bind();

    while (true) {
        //Decode a frame on the GPU using VAAPI
        capture >> videoFrame;
        if (videoFrame.empty()) {
            cerr << "End of stream. Exiting" << endl;
            break;
        }

        //Color-conversion from RGB to BGRA. (OpenCL)
        cv::cvtColor(videoFrame, videoFrameRGBA, cv::COLOR_RGB2BGRA);

        //Activate the OpenCL context for OpenGL
        gl::bind();
        //Initially aquire the framebuffer so we can write the video frame to it
        gl::acquire_from_gl(frameBuffer);
        //Resize the frame if necessary. (OpenCL)
        cv::resize(videoFrameRGBA, frameBuffer, cv::Size(WIDTH, HEIGHT));
        //Release the frame buffer for use by OpenGL
        gl::release_to_gl(frameBuffer);

        //Render using OpenGL
        gl::begin();
        render_scene(WIDTH, HEIGHT);
        gl::end();

        //Aquire the frame buffer for use by OpenCL
        gl::acquire_from_gl(frameBuffer);
        //Glow effect (OpenCL)
        glow_effect(frameBuffer, frameBuffer, GLOW_KERNEL_SIZE);
        //Color-conversion from BGRA to RGB. (OpenCL)
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);
        //Release the frame buffer for use by OpenGL
        gl::release_to_gl(frameBuffer);

        //If x11 is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
        if(!gl::display())
            break;

        //Activate the OpenCL context for VAAPI
        va::bind();
        //Encode the frame using VAAPI on the GPU.
        writer << videoFrame;

        print_fps();
    }

    return 0;
}
