#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

class CustomSourceAndSinkPlan : public Plan {
	string hr_ = "Hello Rainbow!";
public:
	CustomSourceAndSinkPlan(const cv::Size& sz) : Plan(sz) {
	}

	void infer(cv::Ptr<V4D> win) override {
		win->capture();

		//Render "Hello Rainbow!" over the video
		win->nvg([](const Size& sz, const string& str) {
			using namespace cv::v4d::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, str.c_str(), str.c_str() + str.size());
		}, win->fbSize(), hr_);

		win->write();
	}
};

int main() {
	Ptr<CustomSourceAndSinkPlan> plan = new CustomSourceAndSinkPlan(cv::Size(960, 960));
    Ptr<V4D> window = V4D::make(plan->size(), "Custom Source/Sink");

	//Make a source that generates rainbow frames.
	cv::Ptr<Source> src = new Source([](cv::UMat& frame){
		static long cnt = 0;
	    //The source is responsible for initializing the frame..
		if(frame.empty())
		    frame.create(Size(960, 960), CV_8UC3);
	    frame = colorConvert(Scalar(++cnt % 180, 128, 128, 255), COLOR_HLS2BGR);
	    return true;
	}, 60.0f);

	//Make a sink the saves each frame to a PNG file (does nothing in case of WebAssembly).
	cv::Ptr<Sink> sink = new Sink([](const uint64_t& seq, const cv::UMat& frame){
	    try {
			imwrite(std::to_string(seq) + ".png", frame);
	    } catch(std::exception& ex) {
	        cerr << "Unable to write frame: " << ex.what() << endl;
	        return false;
	    }
        return true;
	});

	//Attach source and sink
	window->setSource(src);
	window->setSink(sink);

	window->run(plan);
}

