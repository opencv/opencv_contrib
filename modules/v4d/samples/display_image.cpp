#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

int main() {
    //Creates a V4D window for on screen rendering
    Ptr<V4D> window = V4D::make(Size(1280, 720), cv::Size(), "Display image");

    //An image
#ifdef __EMSCRIPTEN__
    Mat image = read_embedded_image("doc/lena.png");
#else
	Mat image = imread(samples::findFile("lena.jpg"));
#endif
	//Display the framebuffer in the native window in an endless loop.
    //V4D::run() though it takes a functor is not a context. It is simply an abstraction
    //of a run loop for portability reasons and executes the functor until the application
    //terminates or the functor returns false.
	window->run([=](){
	    //Feeds the image to the video pipeline
	    window->feed(image);
	    return window->display();
    });
}
