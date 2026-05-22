#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

class DisplayImageNVG : public Plan {
	//A simple struct to hold our image variables
	struct Image_t {
	    std::string filename_;
	    nvg::Paint paint_;
	    int w_;
	    int h_;
	} image_;
public:
	DisplayImageNVG(const cv::Size& sz) : Plan(sz) {
	}

	void setup(Ptr<V4D> win) override{
		//Set the filename
		image_.filename_ = samples::findFile("lena.jpg");

		//Creates a NanoVG context. The wrapped C-functions of NanoVG are available in the namespace cv::v4d::nvg;
		win->nvg([](Image_t& img) {
			using namespace cv::v4d::nvg;
			//Create the image_ and receive a handle.
			int handle = createImage(img.filename_.c_str(), NVG_IMAGE_NEAREST);
			//Make sure it was created successfully
			CV_Assert(handle > 0);
			//Query the image_ size
			imageSize(handle, &img.w_, &img.h_);
			//Create a simple image_ pattern with the image dimensions
			img.paint_ = imagePattern(0, 0, img.w_, img.h_, 0.0f/180.0f*NVG_PI, handle, 1.0);
		}, image_);
	}

	void infer(Ptr<V4D> win) override{
		//Creates a NanoVG context to draw the loaded image_ over again to the screen.
		win->nvg([](const Image_t& img, const cv::Size& sz) {
			using namespace cv::v4d::nvg;
			beginPath();
			//Scale all further calls to window size
			scale(double(sz.width)/img.w_, double(sz.height)/img.h_);
			//Create a rounded rectangle with the images dimensions.
			//Note that actually this rectangle will have the size of the window
			//because of the previous scale call.
			roundedRect(0,0, img.w_, img.h_, 50);
			//Fill the rounded rectangle with our picture
			fillPaint(img.paint_);
			fill();
		}, image_, win->fbSize());
	}
};

int main() {
    Ptr<DisplayImageNVG> plan = new DisplayImageNVG(cv::Size(960,960));
	Ptr<V4D> window = V4D::make(plan->size(), "Display an Image using NanoVG");
    window->run(plan);
}
