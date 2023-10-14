#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

int main(int argc, char** argv) {
    //In case of WebAssembly
    CV_UNUSED(argc);
    CV_UNUSED(argv);

    Ptr<V4D> window = V4D::make(960, 960, "Video Editing");

    class VideoEditingPlan : public Plan {
    	cv::UMat frame_;
        const string hv_ = "Hello Video!";
    public:
    	void infere(Ptr<V4D> win) override {
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

#ifndef __EMSCRIPTEN__
    //Make the video source
    auto src = makeCaptureSource(window, argv[1]);

    //Make the video sink
    auto sink = makeWriterSink(window, argv[2], src->fps(), cv::Size(960, 960));
    //Attach source and sink
    window->setSource(src);
    window->setSink(sink);
#else
    //Make a webcam Source
    Source src = makeCaptureSource(960, 960, window);
    //Attach webcam source
    window->setSource(src);
#endif

    window->run<VideoEditingPlan>(0);
}

