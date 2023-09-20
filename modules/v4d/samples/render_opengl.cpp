#include <opencv2/v4d/v4d.hpp>


using namespace cv;
using namespace cv::v4d;

int main() {
    Ptr<V4D> window = V4D_INIT_MAIN(960, 960, "GL Blue Screen", false, false, 0);

    window->gl([]() {
        //Sets the clear color to blue
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    });

    window->run([](Ptr<V4D> win) {
        win->gl([]() {
            //Clears the screen. The clear color and other GL-states are preserved between context-calls.
            glClear(GL_COLOR_BUFFER_BIT);
        });

        return win->display();
    });
}

