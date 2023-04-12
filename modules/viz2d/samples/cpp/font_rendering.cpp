#include <opencv2/viz2d/viz2d.hpp>
#include <opencv2/viz2d/nvg.hpp>

using namespace cv;
using namespace cv::viz;

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main(int argc, char** argv) {
	Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Font Rendering");

	//The text to render
	string hw = "Hello World";
	//Clear with black
	v2d->clear();
	//Render the text at the center of the screen
	v2d->nvg([&](const Size& sz) {
		using namespace cv::viz::nvg;
		fontSize(40.0f);
		fontFace("sans-bold");
		fillColor(Scalar(255, 0, 0, 255));
		textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
		text(sz.width / 2.0, sz.height / 2.0, hw.c_str(), hw.c_str() + hw.size());
	});

    //Display the framebuffer in the native window in an endless loop
    v2d->run(v2d->display);
}

