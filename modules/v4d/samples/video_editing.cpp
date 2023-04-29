#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

static Ptr<V4D> window = V4D::make(cv::Size(1280, 720), "Video Editing");

int main(int argc, char** argv) {
    try {
    //In case of emscripten
    CV_UNUSED(argc);
    CV_UNUSED(argv);

    string hv = "Hello Video!";

#ifndef __EMSCRIPTEN__
    //Make the video source
    Source src = makeCaptureSource(argv[1]);

    //Make the video sink
    Sink sink = makeWriterSink(argv[2], VideoWriter::fourcc('V', 'P', '9', '0'), src.fps(), window->getFrameBufferSize());

    //Attach source and sink
    window->setSource(src);
    window->setSink(sink);
#else
    //Make a webcam Source
    Source src = makeCaptureSource(1280,720);
    //Attach web source
    window->setSource(src);
#endif

    window->run([=]() {
	    //Capture video from the Source
		if(!window->capture())
			return false; //end of input video

		window->nvg([=](const Size& sz) {
			using namespace cv::v4d::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, hv.c_str(), hv.c_str() + hv.size());
		});
		updateFps(window,true);

		window->write(); //Write video to the Sink

		return window->display(); //Display the framebuffer in the native window
	});
    } catch(std::exception& ex) {
        cerr << ex.what() << endl;
    }
}

