#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

class RenderOpenGLPlan : public Plan {
public:
	void setup(Ptr<V4D> win) override {
		win->gl([]() {
			//Sets the clear color to blue
			glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
		});
	}
	void infer(Ptr<V4D> win) override {
		win->gl([]() {
			//Clears the screen. The clear color and other GL-states are preserved between context-calls.
			glClear(GL_COLOR_BUFFER_BIT);
		});
	}
};

int main() {
    Ptr<V4D> window = V4D::make(960, 960, "GL Blue Screen");
    window->run<RenderOpenGLPlan>(0);
}

