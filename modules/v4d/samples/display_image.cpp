#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

int main() {
    //Creates a V4D window for on screen rendering with a window size of 720p and a framebuffer of the same size.
	//Please note that while the window size may change the framebuffer size may not. If you need multiple framebuffer
	//sizes you need multiple V4D objects
    Ptr<V4D> window = V4D::make(Size(960, 960), cv::Size(), "Display image");

    //Loads an image as a UMat (just in case we have hardware acceleration available)
#ifdef __EMSCRIPTEN__
    UMat image = read_embedded_image("doc/lena.png").getUMat(ACCESS_READ);
#else
	UMat image = imread(samples::findFile("lena.jpg")).getUMat(ACCESS_READ);;
#endif
	//Display the framebuffer in the native window in an endless loop.
    //V4D::run() though it takes a functor is not a context. It is simply an abstraction
    //of a run loop for portability reasons and executes the functor until the application
    //terminates or the functor returns false.
	window->run([=](Ptr<V4D> window){
	    //Feeds the image to the video pipeline
	    window->feed(image);
	    //Displays the framebuffer in the window. Returns false if the windows has been closed.
	    return window->display();
    });
}
