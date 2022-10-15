#define CL_TARGET_OPENCL_VERSION 220

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr bool OFFSCREEN = false;
constexpr const char* INPUT_FILENAME = "example.mp4";
constexpr const char* OUTPUT_FILENAME = "camera-demo.mkv";
constexpr const int VA_HW_DEVICE_INDEX = 0;
double FPS;

#include "../common/subsystems.hpp"

using std::cerr;
using std::endl;

cv::ocl::OpenCLExecutionContext VA_CONTEXT;
cv::ocl::OpenCLExecutionContext GL_CONTEXT;

void render() {
    //Render a tetrahedron using immediate mode because the code is more concise for a demo
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
    glFlush();
}

void glow(cv::UMat &src, int ksize = WIDTH / 85 % 2 == 0 ? WIDTH / 85  + 1 : WIDTH / 85) {
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
    //Initialize OpenCL Context for VAAPI
    va::init_va();
    /*
     * The OpenCLExecutionContext for VAAPI needs to be copied right after init_va().
     * Now everytime you want to do VAAPI interop first bind the context.
     */
    VA_CONTEXT = cv::ocl::OpenCLExecutionContext::getCurrent();

    //Initialize MJPEG HW decoding using VAAPI
    cv::VideoCapture cap(INPUT_FILENAME, cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    FPS = cap.get(cv::CAP_PROP_FPS);
    std::cerr << "Detected FPS: " << FPS << std::endl;

    cv::VideoWriter video(OUTPUT_FILENAME, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, cv::Size(WIDTH, HEIGHT), {
            cv::VIDEOWRITER_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
            cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    //If we are rendering offscreen we don't need x11
    if(!OFFSCREEN)
        x11::init_x11();

    //Passing true to init_egl will create a OpenGL debug context
    egl::init_egl();
    //Initialize OpenCL Context for OpenGL
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

    cv::UMat frameBuffer;
    cv::UMat videoFrame;
    cv::UMat videoFrameRGBA;

    uint64_t cnt = 1;
    int64 start = cv::getTickCount();
    double tickFreq = cv::getTickFrequency();

    while (true) {
        //Activate the OpenCL context for OpenGL
        GL_CONTEXT.bind();
        //Initially get the framebuffer so we can write the video frame to it
        gl::fetch_frame_buffer(frameBuffer);

        //Activate the OpenCL context for VAAPI
        VA_CONTEXT.bind();
        //Decode a frame using HW acceleration into a cv::UMat
        cap >> videoFrame;
        if (videoFrame.empty()) {
            cerr << "End of stream. Exiting" << endl;
            break;
        }

        //The video is upside-down. Flip it. (OpenCL)
        cv::flip(videoFrame, videoFrame, 0);
        //Color-conversion from BGRA to RGB. (OpenCL)
        cv::cvtColor(videoFrame, videoFrameRGBA, cv::COLOR_RGB2BGRA);
        //Resize the frame. (OpenCL)
        cv::resize(videoFrameRGBA, frameBuffer, cv::Size(WIDTH, HEIGHT));

        GL_CONTEXT.bind();
        //Transfer buffer ownership to OpenGL
        gl::return_frame_buffer(frameBuffer);
        render();
        //Transfer buffer ownership to OpenCL
        gl::fetch_frame_buffer(frameBuffer);

        //Glow effect (OpenCL)
        glow(frameBuffer);

        //Color-conversion from BGRA to RGB. (OpenCL)
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);
        cv::flip(videoFrame, videoFrame, 0);

        //Activate the OpenCL context for VAAPI
        VA_CONTEXT.bind();
        //Encode the frame using VAAPI on the GPU.
        video.write(videoFrame);

        if(x11::is_initialized()) {
            //Yet again activate the OpenCL context for OpenGL
            GL_CONTEXT.bind();
            //Transfer buffer ownership back to OpenGL
            gl::return_frame_buffer(frameBuffer);
            //Blit the framebuffer we have been working on to the screen
            gl::blit_frame_buffer_to_screen();

            //Check if the x11 window was closed
            if(x11::window_closed())
                break;

            gl::swapBuffers();
        }

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
