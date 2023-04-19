#include <opencv2/v4d/v4d.hpp>

int main(int argc, char** argv) {
    using namespace cv;
    using namespace cv::viz;

    Ptr<V4D> v4d = V4D::make(Size(1280, 720), "GL Tetrahedron");

	v4d->gl([](const Size sz) {
#ifndef OPENCV_V4D_ES_VERSION
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
	});

	v4d->run([=]() {
		v4d->gl([](const Size& sz) {
#ifndef OPENCV_V4D_ES_VERSION
			//Render a tetrahedron using immediate mode because the code is more concise
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
#endif
		});

		//If onscreen rendering is enabled it displays the framebuffer in the native window.
		//Returns false if the window was closed.
		return v4d->display();
	});
}
