#include <opencv2/v4d/v4d.hpp>
#include <opencv2/v4d/nvg.hpp>

using namespace cv;
using namespace cv::viz;

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main(int argc, char** argv) {
	string hv = "Hello Video!";
	Ptr<V4D> v2d = V4D::make(Size(WIDTH, HEIGHT), "Video Editing");
	//Make the video source
	Source src = makeCaptureSource(argv[1]);
	//Make the video sink
	Sink sink = makeWriterSink(argv[2], VideoWriter::fourcc('V', 'P', '9', '0'), src.fps(), Size(WIDTH, HEIGHT));

	//Attach source and sink
	v2d->setSource(src);
	v2d->setSink(sink);

	v2d->run([=]() {
	    //Capture video from the Source
		if(!v2d->capture())
			return false; //end of input video

		v2d->nvg([=](const Size& sz) {
			using namespace cv::viz::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, hv.c_str(), hv.c_str() + hv.size());
		});
		v2d->write(); //Write video to the Sink
		return v2d->display(); //Display the framebuffer in the native window
	});
}

