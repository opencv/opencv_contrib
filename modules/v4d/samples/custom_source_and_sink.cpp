#include <opencv2/v4d/v4d.hpp>
#ifndef __EMSCRIPTEN__
#  include <opencv2/imgcodecs.hpp>
#endif

using namespace cv;
using namespace cv::v4d;

static Ptr<V4D> window = V4D::make(Size(1280, 720), cv::Size(), "Custom Source/Sink");

int main() {
    string hr = "Hello Rainbow!";
	//Make a Source that generates rainbow frames.
	Source src([](cv::UMat& frame){
		static long cnt = 0;
	    //The source is responsible for initializing the frame..
		if(frame.empty())
		    frame.create(Size(1280, 720), CV_8UC3);
	    frame = colorConvert(Scalar(++cnt % 180, 128, 128, 255), COLOR_HLS2BGR);
	    return true;
	}, 60.0f);

	//Make a Sink the saves each frame to a PNG file.
	Sink sink([](const cv::UMat& frame){
	    try {
#ifndef __EMSCRIPTEN__
			static long cnt = 0;
			imwrite(std::to_string(cnt++) + ".png", frame);
#else
	        CV_UNUSED(frame);
#endif
	    } catch(std::exception& ex) {
	        cerr << "Unable to write frame: " << ex.what() << endl;
	        return false;
	    }
        return true;
	});

	//Attach source and sink
	window->setSource(src);
	window->setSink(sink);

	window->run([=]() {
	    //Capture video from the Source
		if(!window->capture())
			return false; //end of input video

		//Render "Hello Rainbow!" over the frame
		window->nvg([=](const Size& sz) {
			using namespace cv::v4d::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, hr.c_str(), hr.c_str() + hr.size());
		});

		window->showFps();

		window->write(); //Write video to the Sink
		return window->display(); //Display the framebuffer in the native window
	});
}

