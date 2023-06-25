#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

int main() {
    Ptr<V4D> window = V4D::make(Size(960, 540), cv::Size(), "GL Blue Screen");

    window->gl([](){
        //Sets the clear color to blue
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    });
    window->run([=]() {
        window->gl([]() {
            //Clears the screen. The clear color and other GL-states are preserved between context-calls.
            glClear(GL_COLOR_BUFFER_BIT);
        });

        return window->display();
    });
}

