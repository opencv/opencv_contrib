#define CL_TARGET_OPENCL_VERSION 120

//WIDTH and HEIGHT have to be specified before including subsystems.hpp
constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr const int VA_HW_DEVICE_INDEX = 0;
constexpr bool OFFSCREEN = false;
constexpr const char *OUTPUT_FILENAME = "video-demo.mkv";

#include "../common/subsystems.hpp"
#include <stdlib.h>
#include <string>

using std::cerr;
using std::endl;
using std::string;

void init_tetrahedron() {
    glViewport(0, 0, WIDTH, HEIGHT);
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

void render_tetrahedron() {
    //Render a tetrahedron using immediate mode because the code is more concise for a demo
    glViewport(0, 0, WIDTH, HEIGHT);
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

void glow_effect(cv::UMat &src, int ksize = WIDTH / 85 % 2 == 0 ? WIDTH / 85  + 1 : WIDTH / 85) {
    static cv::UMat resize;
    static cv::UMat blur;
    static cv::UMat src16;

    cv::bitwise_not(src, src);

    //Resize for some extra performance
    cv::resize(src, resize, cv::Size(), 0.5, 0.5);
    //Cheap blur
    cv::boxFilter(resize, resize, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    //Back to original size
    cv::resize(resize, blur, cv::Size(WIDTH, HEIGHT));

    //Multiply the src image with a blurred version of itself
    cv::multiply(src, blur, src16, 1, CV_16U);
    //Normalize and convert back to CV_8U
    cv::divide(src16, cv::Scalar::all(255.0), src, 1, CV_8U);

    cv::bitwise_not(src, src);
}

int main(int argc, char **argv) {
    using namespace kb;

    if(argc != 2) {
        cerr << "Usage: video-demo <video-file>" << endl;
        exit(1);
    }
    //Initialize OpenCL Context for VAAPI
    va::init();

    //Initialize MJPEG HW decoding using VAAPI
    cv::VideoCapture capture(argv[1], cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    if (!capture.isOpened()) {
        cerr << "ERROR! Unable to video input" << endl;
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
        x11::init();

    //Passing true to init_egl will create a OpenGL debug context
    egl::init();
    //Initialize OpenCL Context for OpenGL
    gl::init();

    cerr << "VA Version: " << va::get_info() << endl;
    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    cv::UMat frameBuffer;
    cv::UMat videoFrame;
    cv::UMat videoFrameRGBA;

    uint64_t cnt = 1;
    int64 start = cv::getTickCount();
    double tickFreq = cv::getTickFrequency();
    double lastFps = fps;

    init_tetrahedron();

    //Activate the OpenCL context for VAAPI
    va::bind();

    while (true) {
        //Decode a frame on the GPU using VAAPI
        capture >> videoFrame;
        if (videoFrame.empty()) {
            cerr << "End of stream. Exiting" << endl;
            break;
        }

        //The frameBuffer is upside-down. Flip videoFrame. (OpenCL)
        cv::flip(videoFrame, videoFrame, 0);
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
        render_tetrahedron();
        gl::end();

        //Aquire the frame buffer for use by OpenCL
        gl::acquire_from_gl(frameBuffer);
        //Glow effect (OpenCL)
        glow_effect(frameBuffer);
        //Color-conversion from BGRA to RGB. (OpenCL)
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);
        //Release the frame buffer for use by OpenGL
        gl::release_to_gl(frameBuffer);

        if(x11::is_initialized()) {
            //Blit the framebuffer we have been working on to the screen
            gl::blit_frame_buffer_to_screen();

            //Check if the x11 window was closed
            if(x11::window_closed())
                break;

            //Transfer the back buffer (which we have been using as frame buffer) to the native window
            gl::swap_buffers();
        }

        //Activate the OpenCL context for VAAPI
        va::bind();
        //Video frame is upside down -> flip it (OpenCL)
        cv::flip(videoFrame, videoFrame, 0);
        //Encode the frame using VAAPI on the GPU.
        writer.write(videoFrame);

        //Measure FPS
        if (cnt % uint64(ceil(lastFps)) == 0) {
            int64 tick = cv::getTickCount();
            lastFps = tickFreq / ((tick - start + 1) / cnt);
            cerr << "FPS : " << lastFps << '\r';
            start = tick;
            cnt = 1;
        }

        ++cnt;
    }

    return 0;
}
