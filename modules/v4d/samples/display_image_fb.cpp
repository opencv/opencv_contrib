#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

//Creates a V4D object for on screen rendering
static Ptr<V4D> window = V4D::make(Size(1280, 720), "Show image");

int main() {
	//Read an image as UMat
#ifdef __EMSCRIPTEN__
    UMat image = read_embedded_image("doc/lena.png").getUMat(ACCESS_READ);
#else
    UMat image = imread(samples::findFile("lena.jpg")).getUMat(ACCESS_READ);
#endif
    UMat resized;
	//Resize and color convert the image to framebuffer size
    window->fb([&](const UMat& framebuffer) {
        resize(image, resized, window->getFrameBufferSize());
        cvtColor(resized, framebuffer, COLOR_RGB2BGRA);
    });
	//Display the framebuffer in the native window in an endless loop
	window->run([=](){
	    updateFps(window, true);
		return window->display();
	});
}

