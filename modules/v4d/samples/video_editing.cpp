#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

int main(int argc, char** argv) {
    Ptr<V4D> window = V4D::make(960, 960, "Video Editing", false, false, 0);

    //In case of WebAssembly
    CV_UNUSED(argc);
    CV_UNUSED(argv);

    string hv = "Hello Video!";

#ifndef __EMSCRIPTEN__
    //Make the video source
    Source src = makeCaptureSource(argv[1]);

    //Make the video sink
    Sink sink = makeWriterSink(argv[2], src.fps(), cv::Size(960, 960));
    //Attach source and sink
    window->setSource(src);
    window->setSink(sink);
#else
    //Make a webcam Source
    Source src = makeCaptureSource(960, 960, window);
    //Attach webcam source
    window->setSource(src);
#endif

    window->run([hv](Ptr<V4D> win) {
	    //Capture video from the source
		if(!win->capture())
			return false; //end of input video

		//Render on top of the video
		win->nvg([hv](const Size& sz) {
			using namespace cv::v4d::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, hv.c_str(), hv.c_str() + hv.size());
		});

		//Write video to the sink (do nothing in case of WebAssembly)
		win->write();

		return win->display();
	});
}

