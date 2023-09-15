#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

int main() {
    //Creates a V4D object
    Ptr<V4D> window = V4D::make(Size(960, 960), cv::Size(), "Display image and FB");

    //Loads an image as a UMat (just in case we have hardware acceleration available)
#ifdef __EMSCRIPTEN__
    UMat image = read_embedded_image("doc/lena.png").getUMat(ACCESS_READ);
#else
    UMat image = imread(samples::findFile("lena.jpg")).getUMat(ACCESS_READ);
#endif
    //We have to manually resize and color convert the image when using direct frambuffer access.
    UMat resized;
    UMat converted;
    resize(image, resized, window->framebufferSize());
    cvtColor(resized, converted, COLOR_RGB2BGRA);

    window->run([=](Ptr<V4D> window){
		//Create a fb context and copy the prepared image to the framebuffer. The fb context
		//takes care of retrieving and storing the data on the graphics card (using CL-GL
		//interop if available), ready for other contexts to use
		window->fb([&](UMat& framebuffer){
	        converted.copyTo(framebuffer);
	    });
		return window->display();
	});
}
