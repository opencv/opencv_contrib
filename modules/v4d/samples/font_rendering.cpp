#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

int main() {
    cv::Ptr<V4D> window = V4D_INIT_MAIN(960, 960, "Font Rendering", false, false, 0);

    //The text to render
	string hw = "Hello World";

    window->run([=](Ptr<V4D> window){
        //Render the text at the center of the screen. Note that you can load you own fonts.
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

