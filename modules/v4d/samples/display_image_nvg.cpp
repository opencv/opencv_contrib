#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

//A simple struct to hold our image variables
struct Image_t {
    std::string filename_;
    nvg::Paint paint_;
    int w_;
    int h_;
};

int main() {
    cv::Ptr<V4D> window = V4D::make(960, 960, "Display an Image using NanoVG");
    Image_t image;
    //Set the filename
#ifdef __EMSCRIPTEN__
    image.filename_ = "doc/lena.png";
#else
    image.filename_ = samples::findFile("lena.jpg");
#endif
	//Create the run loop
	window->run([&image](Ptr<V4D> win){
	    //Execute only once
	    win->once([win, &image]() {
	        //Creates a NanoVG context. The wrapped C-functions of NanoVG are available in the namespace cv::v4d::nvg;
	        win->nvg([&image](const cv::Size sz) {
	            using namespace cv::v4d::nvg;
	            //Create the image and receive a handle.
	            int handle = createImage(image.filename_.c_str(), NVG_IMAGE_NEAREST);
	            //Make sure it was created successfully
	            CV_Assert(handle > 0);
	            //Query the image size
	            imageSize(handle, &image.w_, &image.h_);
	            //Create a simple image pattern with the image dimensions
	            image.paint_ = imagePattern(0, 0, image.w_, image.h_, 0.0f/180.0f*NVG_PI, handle, 1.0);
	        });
	    });

	    //Creates a NanoVG context to draw the loaded image over again to the screen.
	    win->nvg([&image](const cv::Size sz) {
	        using namespace cv::v4d::nvg;
	        beginPath();
	        //Scale all further calls to window size
	        scale(double(sz.width)/image.w_, double(sz.height)/image.h_);
	        //Create a rounded rectangle with the images dimensions.
	        //Note that actually this rectangle will have the size of the window
	        //because of the previous scale call.
	        roundedRect(0,0, image.w_, image.h_, 50);
	        //Fill the rounded rectangle with our picture
	        fillPaint(image.paint_);
	        fill();
	    });
	    //Displays the framebuffer in the window. Returns false if the windows has been closed.
	    return win->display();
    });
}
