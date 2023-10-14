#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

int main() {
    //Creates a V4D window for on screen rendering with a window size of 960x960 and a framebuffer of the same size.
	//Please note that while the window size may change the framebuffer size may not. If you need multiple framebuffer
	//sizes you need multiple V4D objects
    cv::Ptr<V4D> window = V4D::make(960, 960, "Display an Image");

    class DisplayImagePlan : public Plan {
    	UMat image_ = imread(samples::findFile("lena.jpg")).getUMat(ACCESS_READ);

    	void setup(Ptr<V4D> win) override {
#ifdef __EMSCRIPTEN__
    		image_ = read_embedded_image("doc/lena.png").getUMat(ACCESS_READ);
#else
    		image_ = imread(samples::findFile("lena.jpg")).getUMat(ACCESS_READ);
#endif
    	}
		//Display the framebuffer in the native window in an endless loop.
		void infer(Ptr<V4D> win) override {
			//Feeds the image to the video pipeline
			win->feed(image_);
		}
    };
}
