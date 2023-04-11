#include <opencv2/viz2d/viz2d.hpp>
#include <opencv2/viz2d/nvg.hpp>

using namespace cv;
using namespace cv::viz;

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main(int argc, char** argv) {
	Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Vector Graphics and Framebuffer");
	//Creates a NanoVG context and draws a cross-hair on the framebuffer
	v2d->nvg([](const Size& sz) {
		//Calls from this namespace may only be used inside a nvg context
		using namespace cv::viz::nvg;
		beginPath();
		strokeWidth(3.0);
		strokeColor(Scalar(0,0,255,255)); //BGRA
		moveTo(WIDTH/2.0, 0);
		lineTo(WIDTH/2.0, HEIGHT);
		moveTo(0, HEIGHT/2.0);
		lineTo(WIDTH, HEIGHT/2.0);
		stroke();
	});

	v2d->fb([](UMat& framebuffer) {
		//Heavily blurs the crosshair using a cheap boxFilter
		boxFilter(framebuffer, framebuffer, -1, Size(15, 15), Point(-1,-1), true, BORDER_REPLICATE);
	});
	while(v2d->display());
}

