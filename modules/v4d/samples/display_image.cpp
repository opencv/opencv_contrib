#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::viz;

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main() {
	//Creates a V4D object for on screen rendering
	Ptr<V4D> v4d = V4D::make(Size(WIDTH, HEIGHT), "Show image");
	//An image
	Mat image = imread(samples::findFile("lena.jpg"));
	//Feeds the image to the video pipeline
	v4d->feed(image);

	//Display the framebuffer in the native window in an endless loop.
        //V4D::run() though it takes a functor is not a context. It is simply an abstraction
        //of a run loop for portability reasons and executes the functor until the application
        //terminates or the functor returns false.
	v4d->run([=](){ return v4d->display(); });
}

