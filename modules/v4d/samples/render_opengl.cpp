#include <opencv2/v4d/v4d.hpp>

int main(int argc, char** argv) {
    using namespace cv;
    using namespace cv::viz;

    Ptr<V4D> v4d = V4D::make(Size(1280, 720), "GL Tetrahedron");
    v4d->setVisible(true);

	v4d->gl([](const Size sz) {
		glViewport(0, 0, sz.width, sz.height);
	});

	v4d->run([=]() {
		v4d->gl([](const Size& sz) {
		    //Clears the screen blue
		    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
		});

		//If onscreen rendering is enabled it displays the framebuffer in the native window.
		//Returns false if the window was closed.
		return v4d->display();
	});
}

