#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

class DisplayImagePlan : public Plan {
	UMat image_;
public:
	DisplayImagePlan(const cv::Size& sz) : Plan(sz) {
	}

	void setup(Ptr<V4D> win) override {
		win->plain([](cv::UMat& image){
			image = imread(samples::findFile("lena.jpg")).getUMat(ACCESS_READ);
		}, image_);
	}

	void infer(Ptr<V4D> win) override {
		//Feeds the image to the video pipeline
		win->feed(image_);
	}
};

int main() {
	cv::Ptr<DisplayImagePlan> plan = new DisplayImagePlan(cv::Size(960,960));
	//Creates a V4D window for on screen rendering with a window size of 960x960 and a framebuffer of the same size.
	//Please note that while the window size may change the framebuffer size may not. If you need multiple framebuffer
	//sizes you need multiple V4D objects
    cv::Ptr<V4D> window = V4D::make(plan->size(), "Display an Image");
    window->run(plan);
}
