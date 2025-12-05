#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

class VideoEditingPlan : public Plan {
	cv::UMat frame_;
	const string hv_ = "Hello Video!";
public:
	VideoEditingPlan(const cv::Size& sz) : Plan(sz) {
	}

	void infer(Ptr<V4D> win) override {
		//Capture video from the source
		win->capture();

		//Render on top of the video
		win->nvg([](const Size& sz, const string& str) {
			using namespace cv::v4d::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, str.c_str(), str.c_str() + str.size());
		}, win->fbSize(), hv_);

		//Write video to the sink (do nothing in case of WebAssembly)
		win->write();
	}
};

int main(int argc, char** argv) {
	if (argc != 3) {
        cerr << "Usage: video_editing <input-video-file> <output-video-file>" << endl;
        exit(1);
    }
    Ptr<VideoEditingPlan> plan = new VideoEditingPlan(cv::Size(960,960));
    Ptr<V4D> window = V4D::make(plan->size(), "Video Editing");

    //Make the video source
    auto src = makeCaptureSource(window, argv[1]);

    //Make the video sink
    auto sink = makeWriterSink(window, argv[2], src->fps(), plan->size());

    //Attach source and sink
    window->setSource(src);
    window->setSink(sink);

    window->run(plan);
}

