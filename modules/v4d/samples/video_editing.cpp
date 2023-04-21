#include <opencv2/v4d/v4d.hpp>

#ifndef __EMSCRIPTEN__
int main(int argc, char** argv) {
#else
int main() {
#endif
    using namespace cv;
    using namespace cv::viz;

    string hv = "Hello Video!";
	Ptr<V4D> v4d = V4D::make(Size(1280, 720), "Video Editing");
    v4d->setVisible(true);
#ifndef __EMSCRIPTEN__
    //Make the video source
    Source src = makeCaptureSource(argv[1]);

	//Make the video sink
	Sink sink = makeWriterSink(argv[2], VideoWriter::fourcc('V', 'P', '9', '0'), src.fps(), v4d->getFrameBufferSize());

    //Attach source and sink
    v4d->setSource(src);
    v4d->setSink(sink);
#else
    //Make a webcam Source
    Source src = makeCaptureSource(1280,720);
    //Attach web source
    v4d->setSource(src);
#endif

	v4d->run([=]() {
	    //Capture video from the Source
		if(!v4d->capture())
			return false; //end of input video

		v4d->nvg([=](const Size& sz) {
			using namespace cv::viz::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, hv.c_str(), hv.c_str() + hv.size());
		});
		v4d->write(); //Write video to the Sink
		return v4d->display(); //Display the framebuffer in the native window
	});
}

