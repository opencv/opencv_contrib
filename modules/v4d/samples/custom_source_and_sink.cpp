#include <opencv2/v4d/v4d.hpp>
#include <opencv2/v4d/nvg.hpp>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char** argv) {
    using namespace cv;
    using namespace cv::viz;

    string hr = "Hello Rainbow!";
	Ptr<V4D> v4d = V4D::make(Size(1280, 720), "Custom Source/Sink");
	//Make a Source that generates rainbow frames.
	Source src([=](cv::UMat& frame){
        static long cnt = 0;

	    if(frame.empty())
	        frame.create(v4d->getFrameBufferSize(), CV_8UC3);
	    frame = colorConvert(Scalar(cnt % 180, 128, 128, 255), COLOR_HLS2BGR);

	    ++cnt;
	    if(cnt > std::numeric_limits<long>().max() / 2.0)
	        cnt = 0;
	    return true;
	}, 60.0f);

	//Make a Sink the saves each frame to a PNG file.
	Sink sink([](const cv::UMat& frame){
        static long cnt = 0;

	    try {
	        imwrite(std::to_string(cnt) + ".png", frame);
	    } catch(std::exception& ex) {
	        cerr << "Unable to write frame: " << ex.what() << endl;
	        return false;
	    }

        ++cnt;
        if(cnt > std::numeric_limits<long>().max() / 2.0)
            cnt = 0;

	    return true;
	});

	//Attach source and sink
	v4d->setSource(src);
	v4d->setSink(sink);

	v4d->run([=]() {
	    //Capture video from the Source
		if(!v4d->capture())
			return false; //end of input video

		//Render "Hello Rainbow!" over the frame
		v4d->nvg([=](const Size& sz) {
			using namespace cv::viz::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, hr.c_str(), hr.c_str() + hr.size());
		});
		v4d->write(); //Write video to the Sink
		return v4d->display(); //Display the framebuffer in the native window
	});
}

