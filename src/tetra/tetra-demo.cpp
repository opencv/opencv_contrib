#define CL_TARGET_OPENCL_VERSION 220

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr double FPS = 30;

#include "subsystems.hpp"

using std::cerr;
using std::endl;

cv::ocl::OpenCLExecutionContext VA_CONTEXT;
cv::ocl::OpenCLExecutionContext GL_CONTEXT;

void render(cv::UMat& frameBuffer) {
    glBindFramebuffer(GL_FRAMEBUFFER, kb::gl::frame_buf);
    glViewport(0, 0, WIDTH , HEIGHT );
    glRotatef(1, 0, 1, 0);
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

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

void blitFrameBufferToScreen() {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, kb::gl::frame_buf);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glViewport(0, 0, WIDTH, HEIGHT);
    glBlitFramebuffer(0, 0, WIDTH, HEIGHT, 0, 0, WIDTH, HEIGHT, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

void glow(cv::UMat &frameBuffer, cv::UMat &mask, double ksize = WIDTH / 90 % 2 == 0 ? WIDTH / 90 + 1 : WIDTH / 90) {
    cv::resize(frameBuffer, mask, cv::Size(), 0.5, 0.5);
    //do the blur on a 50% resized version for some extra performance
    cv::boxFilter(mask, mask, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_CONSTANT);
    cv::resize(mask, mask, cv::Size(WIDTH, HEIGHT));
    cv::bitwise_not(mask, mask);
    cv::bitwise_not(frameBuffer, frameBuffer);
    mask.assignTo(mask, CV_16U);
    frameBuffer.assignTo(frameBuffer, CV_16U);
    cv::multiply(mask, frameBuffer, mask);
    cv::divide(mask, cv::Scalar::all(255.0), mask);
    mask.assignTo(mask, CV_8U);
    cv::bitwise_not(mask, frameBuffer);
}

int main(int argc, char **argv) {
    using namespace kb;

    va::init_va();
    cv::VideoWriter video("tetra-demo.mkv", cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, cv::Size(WIDTH, HEIGHT), { cv::VIDEOWRITER_PROP_HW_DEVICE, 0, cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI, cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1 });
    VA_CONTEXT = cv::ocl::OpenCLExecutionContext::getCurrent();

    //comment the next line for offscreen rendering
    x11::init_x11();

    //Passing true to init_egl will create a OpenGL debug context
    egl::init_egl();
    gl::init_gl();

    GL_CONTEXT = cv::ocl::OpenCLExecutionContext::getCurrent();

    cerr << "VA Version: " << va::get_info() << endl;
    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;


    cv::UMat frameBuffer(HEIGHT, WIDTH, CV_8UC4, cv::Scalar::all(0));
    cv::UMat mask;
    cv::UMat videoFrame;

    int64 start = 0;
    uint64_t cnt = 0;
    while (true) {
        start = cv::getTickCount();

        GL_CONTEXT.bind();

        //Using OpenGL, render a rotating tetrahedron
        render(frameBuffer);

        //Transfer buffer ownership to OpenCL
        cl::fetch_frame_buffer(frameBuffer);

        //Using OpenCL for a glow effect
        glow(frameBuffer, mask);

        //Color-conversion from BGRA to RGB, also OpenCL.
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);

        cv::flip(videoFrame, videoFrame, 0);

        VA_CONTEXT.bind();
        //Encode the frame using VAAPI on the GPU.
        video.write(videoFrame);

        GL_CONTEXT.bind();

        if(x11::is_initialized()) {
            //Transfer buffer ownership back to OpenGL
            cl::return_frame_buffer(frameBuffer);

            //Blit the framebuffer we have been working on to the screen
            blitFrameBufferToScreen();

            //check is the x11 window was closed
            if(x11::window_closed())
                break;
        }

        //Measure FPS
        int64 tick = cv::getTickCount();
        double tickFreq = cv::getTickFrequency();
        if (cnt % int64(ceil(tickFreq / (FPS * 10000000))) == 0) {
            cerr << "FPS : " << tickFreq / (tick - start + 1) << '\r';
            cnt = 0;
        }

        ++cnt;
    }

    return 0;
}
