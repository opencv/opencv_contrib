#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

int main() {
    Ptr<V4D> window = V4D::make(Size(1280, 720), cv::Size(), "Font Rendering");

    //The text to render
	string hw = "Hello World";

    //Display the framebuffer in the native window in an endless loop
    window->run([=](){
        //Render the text at the center of the screen
        window->nvg([&](const Size& sz) {
            using namespace cv::v4d::nvg;
            clear();
            fontSize(40.0f);
            fontFace("sans-bold");
            fillColor(Scalar(255, 0, 0, 255));
            textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
            text(sz.width / 2.0, sz.height / 2.0, hw.c_str(), hw.c_str() + hw.size());
        });

		return window->display();
	});
}

