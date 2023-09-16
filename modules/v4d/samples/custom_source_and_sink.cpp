#include <opencv2/v4d/v4d.hpp>
#ifndef __EMSCRIPTEN__
#  include <opencv2/imgcodecs.hpp>
#endif

using namespace cv;
using namespace cv::v4d;

int main() {
    Ptr<V4D> window = V4D_INIT_MAIN(960, 960, "Custom Source/Sink", false, false, 0);

    string hr = "Hello Rainbow!";
	//Make a source that generates rainbow frames.
	Source src([](cv::UMat& frame){
		static long cnt = 0;
	    //The source is responsible for initializing the frame..
		if(frame.empty())
		    frame.create(Size(1280, 720), CV_8UC3);
	    frame = colorConvert(Scalar(++cnt % 180, 128, 128, 255), COLOR_HLS2BGR);
	    return true;
	}, 60.0f);

	//Make a sink the saves each frame to a PNG file (does nothing in case of WebAssembly).
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

	window->run([=](cv::Ptr<V4D> window) {
		if(!window->capture())
			return false;

		//Render "Hello Rainbow!" over the video
		window->nvg([=](const Size& sz) {
			using namespace cv::v4d::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, hr.c_str(), hr.c_str() + hr.size());
		});

		window->write();

		return window->display();
	});
}

