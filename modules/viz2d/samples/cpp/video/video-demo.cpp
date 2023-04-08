// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "../common/viz2d.hpp"
#include "../common/util.hpp"

#include <string>

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "video-demo.mkv";
constexpr unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

constexpr int GLOW_KERNEL_SIZE = std::max(int(DIAG / 500 % 2 == 0 ? DIAG / 500 + 1 : DIAG / 500), 1);
using std::cerr;
using std::endl;
using std::string;

void init_scene(const cv::Size& sz) {
#ifndef VIZ2D_USE_ES3
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
#endif
}

void render_scene(const cv::Size& sz) {
#ifndef VIZ2D_USE_ES3
    //Render a tetrahedron using immediate mode because the code is more concise for a demo
    glViewport(0, 0, sz.width, sz.height);
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
#endif
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
    using namespace cv::viz;

    if(argc != 2) {
        cerr << "Usage: video-demo <video-file>" << endl;
        exit(1);
    }
    cv::Ptr<Viz2D> v2d = new Viz2D(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Video Demo");
    print_system_info();
    if(!v2d->isOffscreen())
        v2d->setVisible(true);

    Source src = make_capture_source(argv[1]);
    v2d->setSource(src);

    Sink sink = make_writer_sink(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), src.fps(), cv::Size(WIDTH, HEIGHT));
    v2d->setSink(sink);

    v2d->gl(init_scene);

    while (keep_running()) {
        if(!v2d->capture())
            break;

        v2d->gl(render_scene);

        v2d->fb([&](cv::UMat& frameBuffer){
            //Glow effect (OpenCL)
            glow_effect(frameBuffer, frameBuffer, GLOW_KERNEL_SIZE);
        });

        update_fps(v2d, true);

        v2d->write();

        //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
        if(!v2d->display())
            break;
    }

    return 0;
}
