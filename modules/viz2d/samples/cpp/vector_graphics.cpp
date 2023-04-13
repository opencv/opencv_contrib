#include <opencv2/viz2d/viz2d.hpp>
#include <opencv2/viz2d/nvg.hpp>

using namespace cv;
using namespace cv::viz;

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main(int argc, char** argv) {
	Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Vector Graphics");
	//Creates a NanoVG context and draws a cross-hair on the framebuffer
	v2d->nvg([](const Size& sz) {
		//Calls from this namespace may only be used inside a nvg context
		using namespace cv::viz::nvg;

		//Draws a cross hair
		beginPath();
		strokeWidth(3.0);
		strokeColor(Scalar(0,0,255,255)); //BGRA
		moveTo(sz.width/2.0, 0);
		lineTo(sz.width/2.0, sz.height);
		moveTo(0, sz.height/2.0);
		lineTo(sz.width, sz.height/2.0);
		stroke();
	});

    //Display the framebuffer in the native window in an endless loop
    v2d->run(v2d->display);
}

