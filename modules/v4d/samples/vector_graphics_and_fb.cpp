#include <opencv2/v4d/v4d.hpp>
#include <opencv2/v4d/nvg.hpp>

using namespace cv;
using namespace cv::viz;

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main(int argc, char** argv) {
	Ptr<V4D> v4d = V4D::make(Size(WIDTH, HEIGHT), "Vector Graphics and Framebuffer");
	//Creates a NanoVG context and draws a cross-hair on the framebuffer
	v4d->nvg([](const Size& sz) {
		//Calls from this namespace may only be used inside a nvg context
		using namespace cv::viz::nvg;

		//Draws a cross-hair
		beginPath();
		strokeWidth(3.0);
		strokeColor(Scalar(0,0,255,255)); //BGRA
        moveTo(sz.width/2.0, 0);
        lineTo(sz.width/2.0, sz.height);
        moveTo(0, sz.height/2.0);
        lineTo(sz.width, sz.height/2.0);
		stroke();
	});

	v4d->fb([](UMat& framebuffer) {
		//Heavily blurs the crosshair using a cheap boxFilter
		boxFilter(framebuffer, framebuffer, -1, Size(15, 15), Point(-1,-1), true, BORDER_REPLICATE);
	});
        //Display the framebuffer in the native window in an endless loop
        v4d->run([=](){ return v4d->display(); });
}

