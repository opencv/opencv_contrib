#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

class RenderOpenGLPlan : public Plan {
public:
	RenderOpenGLPlan(const cv::Size& sz) : Plan(sz) {
	}

	void setup(Ptr<V4D> window) override {
		window->gl([]() {
			//Sets the clear color to blue
			glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
		});
	}
	void infer(Ptr<V4D> window) override {
		window->gl([]() {
			//Clears the screen. The clear color and other GL-states are preserved between context-calls.
			glClear(GL_COLOR_BUFFER_BIT);
		});
	}
};

int main() {
	Ptr<RenderOpenGLPlan> plan = new RenderOpenGLPlan(cv::Size(960, 960));
    Ptr<V4D> window = V4D::make(plan->size(), "GL Blue Screen");
    window->run(plan);
}

