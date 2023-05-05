#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

static Ptr<V4D> window = V4D::make(Size(1280, 720), "GL Blue Screen");

int main() {
    window->gl([](){
        //Sets blue as clear color
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    });
    window->run([=]() {
        window->gl([]() {
		    //Clears the screen
			glClear(GL_COLOR_BUFFER_BIT);
		});
        window->updateFps();

		//If onscreen rendering is enabled it displays the framebuffer in the native window.
		//Returns false if the window was closed.
		return window->display();
	});
}
