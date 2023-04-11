#include <opencv2/viz2d/viz2d.hpp>

using namespace cv;
using namespace cv::viz;

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main(int argc, char** argv) {
	Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "GL Tetrahedron");

	v2d->gl([](const Size sz) {
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
	});
	//Viz2D::run() though it takes a functor is not a context. It is simply an abstraction
	//of a run loop for portability reasons and executes the functor until the application
	//terminates or the functor returns false.
	v2d->run([=]() {
		v2d->gl([](const Size& sz) {
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
		});

		//If onscreen rendering is enabled it displays the framebuffer in the native window.
		//Returns false if the window was closed.
		return v2d->display();
	});
}

