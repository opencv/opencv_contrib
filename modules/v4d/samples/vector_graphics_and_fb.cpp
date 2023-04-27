#include <opencv2/v4d/v4d.hpp>

int main() {
    using namespace cv;
    using namespace cv::v4d;

	Ptr<V4D> v4d = V4D::make(Size(1280, 720), "Vector Graphics and Framebuffer");
    v4d->setVisible(true);

    //Display the framebuffer in the native window in an endless loop
	v4d->run([=](){
	    //Creates a NanoVG context and draws a cross-hair on the framebuffer
	    v4d->nvg([](const Size& sz) {
	        //Calls from this namespace may only be used inside a nvg context
	        using namespace cv::v4d::nvg;

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
		updateFps(v4d,true);
		return v4d->display();
	});
}

