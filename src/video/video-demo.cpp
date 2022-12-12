
#define CL_TARGET_OPENCL_VERSION 120

#include "../common/viz2d.hpp"
#include <string>

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr const int VA_HW_DEVICE_INDEX = 0;
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "video-demo.mkv";
constexpr unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

constexpr int glow_kernel_size = std::max(int(DIAG / 138 % 2 == 0 ? DIAG / 138 + 1 : DIAG / 138), 1);

using std::cerr;
using std::endl;
using std::string;

void init_scene(unsigned long w, unsigned long h) {
    //Initialize the OpenGL scene
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
    cv::Ptr<kb::Viz2D> window = new kb::Viz2D(cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Video Demo");
    window->initialize();

    if(!window->isOffscreen())
        window->setVisible(true);

    //Print system information
    kb::print_system_info();

    auto capture = window->makeVACapture(argv[1], VA_HW_DEVICE_INDEX);

    if (!capture.isOpened()) {
        cerr << "ERROR! Unable to open video input" << endl;
        exit(-1);
    }

    float fps = capture.get(cv::CAP_PROP_FPS);
    float width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    float height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    window->makeVAWriter(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, {width, height}, VA_HW_DEVICE_INDEX);

    window->render([](const cv::Size& sz) {
        init_scene(sz.width, sz.height);
    });

    while (true) {
        if(!window->captureVA())
            break;

        window->render([](const cv::Size& sz) {
            //Render using OpenGL
            render_scene(sz.width, sz.height);
        });

        window->compute([&](cv::UMat& frameBuffer){
            //Glow effect (OpenCL)
            glow_effect(frameBuffer, frameBuffer, glow_kernel_size);
        });


        window->writeVA();

        //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
        if(!window->display())
            break;
    }

    return 0;
}
