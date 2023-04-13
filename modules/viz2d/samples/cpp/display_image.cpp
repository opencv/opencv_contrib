#include <opencv2/viz2d/viz2d.hpp>

using namespace cv;
using namespace cv::viz;

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main() {
	//Creates a Viz2D object for on screen rendering
	Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Show image");
	//An image
	Mat image = imread(samples::findFile("lena.jpg"));
	//Feeds the image to the video pipeline
	v2d->feed(image);

	//Display the framebuffer in the native window in an endless loop.
    //Viz2D::run() though it takes a functor is not a context. It is simply an abstraction
    //of a run loop for portability reasons and executes the functor until the application
    //terminates or the functor returns false.
	v2d->run(v2d->display);
}

