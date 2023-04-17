#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

int main() {
    using namespace cv;
    using namespace cv::viz;

    //Creates a V4D object for on screen rendering
	Ptr<V4D> v4d = V4D::make(Size(1280, 720), "Show image");
	//Read an image as UMat
	UMat image = imread(samples::findFile("lena.jpg")).getUMat(ACCESS_READ);
	UMat resized;
	//Resize the image to framebuffer size
	resize(image, resized, v4d->getFrameBufferSize());
	v4d->fb([&](const UMat& framebuffer) {
		//Color convert the resized UMat. The framebuffer has alpha.
		cvtColor(resized, framebuffer, COLOR_RGB2BGRA);
	});
	//Display the framebuffer in the native window in an endless loop
	v4d->run([=](){ return v4d->display(); });
}

