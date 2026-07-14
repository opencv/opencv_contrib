#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

class DisplayImageFB : public Plan {
	UMat image_;
	UMat converted_;
public:
	DisplayImageFB(const cv::Size& sz) : Plan(sz) {
	}

	void setup(cv::Ptr<V4D> win) override {
		win->plain([](cv::UMat& image, cv::UMat& converted, const cv::Size& sz){
			//Loads an image as a UMat (just in case we have hardware acceleration available)
			image = imread(samples::findFile("lena.jpg")).getUMat(ACCESS_READ);

			//We have to manually resize and color convert the image when using direct frambuffer access.
			resize(image, converted, sz);
			cvtColor(converted, converted, COLOR_RGB2BGRA);
		}, image_, converted_, win->fbSize());
	}

	void infer(Ptr<V4D> win) override {
		//Create a fb context and copy the prepared image to the framebuffer. The fb context
		//takes care of retrieving and storing the data on the graphics card (using CL-GL
		//interop if available), ready for other contexts to use
		win->fb([](UMat& framebuffer, const cv::UMat& c){
			c.copyTo(framebuffer);
		}, converted_);
	}
};

int main() {
	Ptr<DisplayImageFB> plan = new DisplayImageFB(cv::Size(960,960));
	//Creates a V4D object
    Ptr<V4D> window = V4D::make(plan->size(), "Display an Image through direct FB access");
    window->run(plan);
}
