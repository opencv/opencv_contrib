#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

static Ptr<V4D> window = V4D::make(Size(1280, 720), "Vector Graphics");

int main() {
    //Display the framebuffer in the native window in an endless loop
    window->run([=](){
        window->clear();
        //Creates a NanoVG context and draws a cross-hair on the framebuffer
        window->nvg([](const Size& sz) {
            //Calls from this namespace may only be used inside a nvg context
            using namespace cv::v4d::nvg;

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

		updateFps(window,true);
		return window->display();
	});
}

