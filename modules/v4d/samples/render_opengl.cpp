#include <opencv2/v4d/v4d.hpp>

int main() {
    using namespace cv;
    using namespace cv::v4d;

    Ptr<V4D> v4d = V4D::make(Size(1280, 720), "GL Blue Screen");
    v4d->setVisible(true);

	v4d->run([=]() {
		v4d->gl([]() {
		    //Clears the screen blue
		    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
		});

		//If onscreen rendering is enabled it displays the framebuffer in the native window.
		//Returns false if the window was closed.
		return v4d->display();
	});
}

