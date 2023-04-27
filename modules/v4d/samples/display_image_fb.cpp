#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

int main() {
    using namespace cv;
    using namespace cv::v4d;

    //Creates a V4D object for on screen rendering
	Ptr<V4D> v4d = V4D::make(Size(1280, 720), "Show image");
    v4d->setVisible(true);
	//Read an image as UMat
#ifdef __EMSCRIPTEN__
    UMat image = read_image("doc/lena.png").getUMat(ACCESS_READ);
#else
    UMat image = imread(samples::findFile("lena.jpg")).getUMat(ACCESS_READ);
#endif
    UMat resized;
	//Resize and color convert the image to framebuffer size
    v4d->fb([&](const UMat& framebuffer) {
        resize(image, resized, v4d->getFrameBufferSize());
        cvtColor(resized, framebuffer, COLOR_RGB2BGRA);
    });
	//Display the framebuffer in the native window in an endless loop
	v4d->run([=](){
		return v4d->display();
	});
}

