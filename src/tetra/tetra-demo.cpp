#define CL_TARGET_OPENCL_VERSION 120

#include "../common/subsystems.hpp"

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr double FPS = 60;
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "tetra-demo.mkv";
constexpr const int VA_HW_DEVICE_INDEX = 0;
constexpr unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

constexpr int glow_kernel_size = std::max(int(DIAG / 138 % 2 == 0 ? DIAG / 138 + 1 : DIAG / 138), 1);

using std::cerr;
using std::endl;

void init_scene(unsigned long w, unsigned long h) {
    glViewport(0, 0, w, h);
    //Initialize the OpenGL scene
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
    glViewport(0, 0, w, h);
    //Render a tetrahedron using immediate mode because the code is more concise for a demo
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

    //Initialize the application
    app::init("Tetra Demo", WIDTH, HEIGHT, WIDTH / 2.0, HEIGHT / 2.0, OFFSCREEN);
    //Print system information
    app::print_system_info();

    app::run([&]() {
        cv::Size frameBufferSize(app::frame_buffer_width, app::frame_buffer_height);

        //Initialize VP9 HW encoding using VAAPI
        cv::VideoWriter writer(OUTPUT_FILENAME, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, frameBufferSize, {
                cv::VIDEOWRITER_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
                cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
                cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
        });

        //Copy OpenCL Context for VAAPI. Must be called right after first VideoWriter/VideoCapture initialization.
        va::copy();

        gl::render([](int w, int h) {
            //Initialize the OpenGL scene
            init_scene(w, h);
        });

        while (true) {
            //Render using OpenGL
            gl::render([](int w, int h) {
                render_scene(w, h);
            });

            //Aquire the frame buffer for use by OpenCL
            cl::compute([](cv::UMat &frameBuffer) {
                imshow("fb", frameBuffer);
                cv::waitKey(1);
                //Glow effect (OpenCL)
                glow_effect(frameBuffer, frameBuffer, glow_kernel_size);
            });

            va::write([&writer](const cv::UMat& videoFrame){
                //videoFrame is the frameBuffer converted to BGR. Ready to be written.
                writer << videoFrame;
            });

            app::update_fps();

            //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
            if(!app::display())
                break;
        }
    });

    app::terminate();

    return 0;
}
