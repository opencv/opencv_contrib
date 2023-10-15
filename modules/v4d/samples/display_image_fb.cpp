#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

int main() {
    //Creates a V4D object
    Ptr<V4D> window = V4D::make(960, 960, "Display an Image through direct FB access");

    class DisplayImageFB : public Plan {
		UMat image_;
		UMat resized_;
		UMat converted_;
public:
		void setup(cv::Ptr<V4D> win) {
			win->parallel([](cv::UMat& image, cv::UMat& resized, cv::UMat& converted, const cv::Size& sz){
				//Loads an image as a UMat (just in case we have hardware acceleration available)
#ifdef __EMSCRIPTEN__
				image = read_embedded_image("doc/lena.png").getUMat(ACCESS_READ);
#else
				image = imread(samples::findFile("lena.jpg")).getUMat(ACCESS_READ);
#endif
			    //We have to manually resize and color convert the image when using direct frambuffer access.
				resize(image, resized, sz);
				cvtColor(resized, converted, COLOR_RGB2BGRA);
			}, image_, resized_, converted_, win->fbSize());
		}

		void infer(Ptr<V4D> win) override {
			//Create a fb context and copy the prepared image to the framebuffer. The fb context
			//takes care of retrieving and storing the data on the graphics card (using CL-GL
			//interop if available), ready for other contexts to use
			win->fb([](UMat& framebuffer, const cv::UMat& c){
				c.copyTo(framebuffer);
			}, converted_);
		}
    };

    window->run<DisplayImageFB>(0);
}
