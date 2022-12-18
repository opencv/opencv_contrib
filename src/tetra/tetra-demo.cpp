#define CL_TARGET_OPENCL_VERSION 120

#include "../common/viz2d.hpp"
#include "../common/util.hpp"

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

void init_scene(const cv::Size& sz) {
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
}

void render_scene(const cv::Size& sz) {
    //Render a tetrahedron using immediate mode because the code is more concise for a demo
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
    using namespace kb::viz2d;

    cv::Ptr<Viz2D> v2d = new Viz2D(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Tetra Demo");
    print_system_info();
    if(!v2d->isOffscreen())
        v2d->setVisible(true);

    v2d->makeVAWriter(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, v2d->getFrameBufferSize(), 0);

    v2d->opengl(init_scene);

    while (true) {
        //Render using OpenGL
        v2d->opengl(render_scene);

        //Aquire the frame buffer for use by OpenCL
        v2d->opencl([](cv::UMat &frameBuffer) {
            //Glow effect (OpenCL)
            glow_effect(frameBuffer, frameBuffer, glow_kernel_size);
        });

        update_fps(v2d, true);

        v2d->write();

        //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
        if (!v2d->display())
            break;
    }

    return 0;
}
