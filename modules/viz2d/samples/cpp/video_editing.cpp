#include <opencv2/viz2d/viz2d.hpp>
#include <opencv2/viz2d/nvg.hpp>

using namespace cv;
using namespace cv::viz;

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main(int argc, char** argv) {
	string hv = "Hello Video!";
	Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Video Editing");
	v2d->setVisible(true);
	//Make the video source
	Source src = makeCaptureSource(argv[1]);
	//Make the video sink
	Sink sink = makeWriterSink(argv[2], VideoWriter::fourcc('V', 'P', '9', '0'), src.fps(), Size(WIDTH, HEIGHT));

	//Attach source and sink
	v2d->setSource(src);
	v2d->setSink(sink);

	v2d->run([=]() {
		if(!v2d->capture())
			return false;
		v2d->nvg([=](const Size& sz) {
			using namespace cv::viz::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(WIDTH / 2.0, HEIGHT / 2.0, hv.c_str(), hv.c_str() + hv.size());
		});
		v2d->write();
		return v2d->display();
	});
}

