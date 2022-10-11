#define CL_TARGET_OPENCL_VERSION 220

const long unsigned int WIDTH = 1920;
const long unsigned int HEIGHT = 1080;
double FPS;
constexpr double OFFSCREEN = false;

#include "../tetra/subsystems.hpp"

using std::cerr;
using std::endl;

cv::ocl::OpenCLExecutionContext VA_CONTEXT;
cv::ocl::OpenCLExecutionContext GL_CONTEXT;

void render(cv::UMat& frameBuffer) {
    glBindFramebuffer(GL_FRAMEBUFFER, kb::gl::frame_buf);
    glViewport(0, 0, WIDTH , HEIGHT );
    glRotatef(1, 0, 1, 0);
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);

    glColor3f(1.0, 1.0, 1.0);
    glBegin(GL_LINES);
    for (GLfloat i = -2.5; i <= 2.5; i += 0.25) {
        glVertex3f(i, 0, 2.5);
        glVertex3f(i, 0, -2.5);
        glVertex3f(2.5, 0, i);
        glVertex3f(-2.5, 0, i);
    }
    glEnd();

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
    kb::gl::swapBuffers();
}

void blit_frame_buffer_to_screen() {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, kb::gl::frame_buf);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glViewport(0, 0, WIDTH, HEIGHT);
    glBlitFramebuffer(0, 0, WIDTH, HEIGHT, 0, 0, WIDTH, HEIGHT, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

void bloom(cv::UMat &frameBuffer, int ksize = WIDTH / 200 % 2 == 0 ? WIDTH / 200  + 1 : WIDTH / 200, int thresh_value = 245, int gain = 6) {
    static cv::UMat hsv;
    static cv::UMat s;
    static cv::UMat v;
    static cv::UMat sv;
    static cv::UMat mask;

    static std::vector<cv::UMat> hsvChannels;
    cv::cvtColor(frameBuffer, hsv, cv::COLOR_BGR2HSV);
    cv::split(hsv, hsvChannels);
    cv::bitwise_not(hsvChannels[1], hsvChannels[1]);

    hsvChannels[1].assignTo(s, CV_16U);
    hsvChannels[2].assignTo(v, CV_16U);

    cv::multiply(s, v, sv);
    cv::divide(sv, cv::Scalar(255.0), sv);
    sv.assignTo(sv, CV_8U);

    cv::threshold(sv, sv, thresh_value, 255, cv::THRESH_BINARY);

    cv::resize(sv, sv, cv::Size(), 0.5, 0.5);
    cv::boxFilter(sv, sv, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    cv::resize(sv, sv, cv::Size(WIDTH, HEIGHT));
    cv::cvtColor(sv, mask, cv::COLOR_GRAY2BGRA);

    addWeighted(frameBuffer, 1.0, mask, gain, 0, frameBuffer);
}

int main(int argc, char **argv) {
    using namespace kb;

    va::init_va();
    /*
     * The OpenCLExecutionContext for VAAPI needs to be copied right after init_va().
     * Now everytime you want to do VAAPI interop first bind the context.
     */
    VA_CONTEXT = cv::ocl::OpenCLExecutionContext::getCurrent();

    //Initialize VP9 HW encoding using VAAPI
    cv::VideoCapture cap("output.mjpeg", cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, 0,
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    FPS = cap.get(cv::CAP_PROP_FPS);
    std::cerr << "FPS: " << FPS << std::endl;
    cv::VideoWriter video("camera-demo.mkv", cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, cv::Size(WIDTH, HEIGHT), { cv::VIDEOWRITER_PROP_HW_DEVICE, 0, cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI, cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1 });

    //If we are rendering offscreen we don't need x11
    if(!OFFSCREEN)
        x11::init_x11();

    //Passing true to init_egl will create a OpenGL debug context
    egl::init_egl();
    gl::init_gl();
    /*
     * The OpenCLExecutionContext for OpenGL needs to be copied right after init_gl().
     * Now everytime you want to do OpenGL interop first bind the context.
     */
    GL_CONTEXT = cv::ocl::OpenCLExecutionContext::getCurrent();

    cerr << "VA Version: " << va::get_info() << endl;
    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    cv::UMat frameBuffer(HEIGHT, WIDTH, CV_8UC4, cv::Scalar::all(0));
    cv::UMat videoFrame;

    uint64_t cnt = 1;
    int64 start = cv::getTickCount();
    double tickFreq = cv::getTickFrequency();

    while (true) {
        GL_CONTEXT.bind();
        gl::swapBuffers();
        gl::fetch_frame_buffer(frameBuffer);

        cap >> videoFrame;
        if (videoFrame.empty()) {
            cerr << "End of stream. Exiting" << endl;
            break;
        }
/*
        cv::flip(videoFrame, videoFrame, 0);
        cv::cvtColor(videoFrame, frameBuffer, cv::COLOR_RGB2BGRA);
        cv::resize(frameBuffer, frameBuffer, cv::Size(WIDTH, HEIGHT));
        gl::return_frame_buffer(frameBuffer);
        //Using OpenGL, render a rotating tetrahedron
        render(frameBuffer);
        //Transfer buffer ownership to OpenCL
        gl::fetch_frame_buffer(frameBuffer);
        //Using OpenCV/OpenCL for a glow effect
        bloom(frameBuffer);
        //Color-conversion from BGRA to RGB. Also OpenCV/OpenCL.
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);

        VA_CONTEXT.bind();
        cv::flip(videoFrame, videoFrame, 0);
        //Encode the frame using VAAPI on the GPU.
        video.write(videoFrame);

        if(x11::is_initialized()) {
            GL_CONTEXT.bind();
            //Transfer buffer ownership back to OpenGL
            gl::return_frame_buffer(frameBuffer);
            //Blit the framebuffer we have been working on to the screen
            blit_frame_buffer_to_screen();
            //Check if the x11 window was closed
            if(x11::window_closed())
                break;
        }
*/

        //Measure FPS
        if (cnt % uint64(FPS) == 0) {
            int64 tick = cv::getTickCount();
            cerr << "FPS : " << tickFreq / ((tick - start + 1) / cnt) << '\r';
            start = tick;
            cnt = 1;
        }

        ++cnt;
    }

    return 0;
}
