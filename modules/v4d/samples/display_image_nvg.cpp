#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

struct Image_t {
    std::string filename_;
    cv::v4d::nvg::Paint paint_;
    int w_;
    int h_;
};

int main() {
    cv::Ptr<V4D> window = V4D::make(960, 960, "Display Image using NanoVG", false, false, 0);

#ifdef __EMSCRIPTEN__
    string filename = "doc/lena.png";
#else
    string filename = samples::findFile("lena.jpg");
#endif
    Image_t image;

	window->nvg([&image](const cv::Size sz) {
	    using namespace cv::v4d::nvg;
	    int res = createImage(image.filename_.c_str(), NVG_IMAGE_NEAREST);
	    CV_Assert(res > 0);
	    imageSize(res, &image.w_, &image.h_);
	    image.paint_ = imagePattern(0, 0, image.w_, image.h_, 0.0f/180.0f*NVG_PI, res, 1.0);
	});

	window->run([&image](Ptr<V4D> win){
	    win->nvg([&image](const cv::Size sz) {
	        using namespace cv::v4d::nvg;
	        beginPath();
	        scale(double(sz.width)/image.w_, double(sz.height)/image.h_);
	        roundedRect(0,0, image.w_, image.h_, 50);
	        fillPaint(image.paint_);
	        fill();
	    });
	    //Displays the framebuffer in the window. Returns false if the windows has been closed.
	    return win->display();
    });
}
