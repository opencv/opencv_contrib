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

    cv::Ptr<kb::Window> window = new kb::Window(cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Tetra Demo");

    //Print system information
    kb::print_system_info();

    window->makeVAWriter(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, window->getSize(), 0);

    window->render([](const cv::Size &size) {
        //Initialize the OpenGL scene
        init_scene(size.width, size.height);
    });

    while (true) {
        //Render using OpenGL
        window->render([](const cv::Size &size) {
            render_scene(size.width, size.height);
        });

        //Aquire the frame buffer for use by OpenCL
        window->compute([](cv::UMat &frameBuffer) {
            //Glow effect (OpenCL)
            glow_effect(frameBuffer, frameBuffer, glow_kernel_size);
        });

        window->writeVA();

        update_fps(window);

        //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
        if (!window->display())
            break;
    }

    window->terminate();

    return 0;
}
