#include <opencv2/viz2d/viz2d.hpp>

using namespace cv;
using namespace cv::viz;

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main(int argc, char** argv) {
	Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Manipulate Framebuffer");
	//An image
	Mat image = imread(argv[1]);
	//Feeds the image to the video pipeline
	v2d->feed(image);
	//Directly access the framebuffer using OpenCV
	v2d->fb([](UMat& framebuffer) {
		flip(framebuffer,framebuffer,0); //Flip the framebuffer
	});
	//Display the upside-down image in the native window
	while(v2d->display());
}

