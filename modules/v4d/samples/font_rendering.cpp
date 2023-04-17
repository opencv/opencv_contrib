#include <opencv2/v4d/v4d.hpp>
#include <opencv2/v4d/nvg.hpp>

int main(int argc, char** argv) {
    using namespace cv;
    using namespace cv::viz;

    Ptr<V4D> v4d = V4D::make(Size(1280, 720), "Font Rendering");

	//The text to render
	string hw = "Hello World";
	//Clear with black
	v4d->clear();
	//Render the text at the center of the screen
	v4d->nvg([&](const Size& sz) {
		using namespace cv::viz::nvg;
		fontSize(40.0f);
		fontFace("sans-bold");
		fillColor(Scalar(255, 0, 0, 255));
		textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
		text(sz.width / 2.0, sz.height / 2.0, hw.c_str(), hw.c_str() + hw.size());
	});

    //Display the framebuffer in the native window in an endless loop
    v4d->run([=](){ return v4d->display(); });
}

