#include <opencv2/v4d/v4d.hpp>
#include "../src/detail/framebuffercontext.hpp"

using namespace cv;
using namespace cv::v4d;

int main() {
    Ptr<V4D> window = V4D::make(Size(1280, 720), cv::Size(), "GL Blue Screen");

    window->gl([](){
        //Sets the clear color to blue
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    });
    window->run([=]() {
        window->gl([]() {
            glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
            //Clears the screen
            glClear(GL_COLOR_BUFFER_BIT);
        });

        //If onscreen rendering is enabled it displays the framebuffer in the native window.
        //Returns false if the window was closed.
        return window->display();
    });
}

