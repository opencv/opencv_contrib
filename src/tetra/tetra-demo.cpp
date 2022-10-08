#define CL_TARGET_OPENCL_VERSION 220

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr double FPS = 30;

#include "subsystems.hpp"

#include <sstream>

using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char **argv) {
    using namespace kb;

    va::init_va();
    cv::VideoWriter video("tetra-demo.mkv", cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, cv::Size(WIDTH, HEIGHT), { cv::VIDEOWRITER_PROP_HW_DEVICE, 0, cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI, cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1 });
    cv::ocl::OpenCLExecutionContext vaContext = cv::ocl::OpenCLExecutionContext::getCurrent();

    egl::init_egl();
    gl::init_gl();

    cv::ocl::OpenCLExecutionContext glContext = cv::ocl::OpenCLExecutionContext::getCurrent();

    cerr << "VA Version: " << va::get_info() << endl;
    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    cv::UMat frameBuffer(HEIGHT, WIDTH, CV_8UC4, cv::Scalar::all(0));
    cv::UMat mask;
    cv::UMat videoFrame;

    double sigma = 50;
    int64 start = 0;
    uint64_t cnt = 0;

    while (true) {
        start = cv::getTickCount();

        //Draw a rotating tetrahedron
        glContext.bind();
        glRotatef(1, 0, 1, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

        glFlush();
        gl::swapBuffers();
        cl::fetch_frame_buffer(frameBuffer); //hand over the data (GPU 2 GPU) to OpenCV/OpenCL

        //Using OpenCL in the background
        cv::flip(frameBuffer, frameBuffer, 0); //  flip the image in the y-axis

        {
            //Do a glow effect using blur
            cv::blur(frameBuffer, mask, cv::Size(sigma, sigma));
            cv::bitwise_not(mask, mask);
            cv::bitwise_not(frameBuffer, frameBuffer);
            mask.assignTo(mask, CV_16U);
            frameBuffer.assignTo(frameBuffer, CV_16U);
            cv::multiply(mask, frameBuffer, mask);
            cv::divide(mask, cv::Scalar::all(255.0), mask);
            mask.assignTo(mask, CV_8U);
            cv::bitwise_not(mask, frameBuffer);
        }

        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB); // Color-conversion from BGRA to RGB

        vaContext.bind();
        video.write(videoFrame); //encode the frame using VAAPI on the GPU.

        int64 tick = cv::getTickCount();
        double tickFreq = cv::getTickFrequency();
        if (cnt % int64(ceil(tickFreq / (FPS * 10000000))) == 0)
            cerr << "FPS : " << tickFreq / (tick - start + 1) << '\r';

        ++cnt;
    }

    return 0;
}
