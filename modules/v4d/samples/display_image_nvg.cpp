#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;
using namespace cv::v4d::nvg;

int main() {
    cv::Ptr<V4D> window = V4D::make(960, 960, "Display Image using NanoVG", false, false, 0);

#ifdef __EMSCRIPTEN__
    string filename = "doc/lena.png";
#else
    string filename = samples::findFile("lena.jpg");
#endif
	Paint imagePaint;
    int w, h;
	window->nvg([&imagePaint, &w, &h, filename](const cv::Size sz) {
	    int img = createImage(filename.c_str(), NVG_IMAGE_NEAREST);
	    CV_Assert(img > 0);
	    imageSize(img, &w, &h);
	    imagePaint = imagePattern(0, 0, w, h, 0.0f/180.0f*NVG_PI, img, 1.0);
	});

	window->run([&imagePaint, w, h](Ptr<V4D> win){
	    win->nvg([&imagePaint, w, h](const cv::Size sz) {
	        beginPath();
	        scale(double(sz.width)/w, double(sz.height)/h);
	        roundedRect(0,0, w, h, 5);
	        fillPaint(imagePaint);
	        fill();
	    });
	    //Displays the framebuffer in the window. Returns false if the windows has been closed.
	    return win->display();
    });
}
