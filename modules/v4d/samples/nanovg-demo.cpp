// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

static void draw_color_wheel(float x, float y, float w, float h, double hue) {
    //color wheel drawing code taken from https://github.com/memononen/nanovg/blob/master/example/demo.c
    using namespace cv::v4d::nvg;
    int i;
    float r0, r1, ax, ay, bx, by, cx, cy, aeps, r;
    Paint paint;

    save();

    cx = x + w * 0.5f;
    cy = y + h * 0.5f;
    r1 = (w < h ? w : h) * 0.5f - 5.0f;
    r0 = r1 - 20.0f;
    aeps = 0.5f / r1;   // half a pixel arc length in radians (2pi cancels out).

    for (i = 0; i < 6; i++) {
        float a0 = (float) i / 6.0f * CV_PI * 2.0f - aeps;
        float a1 = (float) (i + 1.0f) / 6.0f * CV_PI * 2.0f + aeps;
        beginPath();
        arc(cx, cy, r0, a0, a1, NVG_CW);
        arc(cx, cy, r1, a1, a0, NVG_CCW);
        closePath();
        ax = cx + cosf(a0) * (r0 + r1) * 0.5f;
        ay = cy + sinf(a0) * (r0 + r1) * 0.5f;
        bx = cx + cosf(a1) * (r0 + r1) * 0.5f;
        by = cy + sinf(a1) * (r0 + r1) * 0.5f;
        paint = linearGradient(ax, ay, bx, by,
                cv::v4d::colorConvert(cv::Scalar((a0 / (CV_PI * 2.0)) * 180.0, 0.55 * 255.0, 255.0, 255.0), cv::COLOR_HLS2BGR),
                cv::v4d::colorConvert(cv::Scalar((a1 / (CV_PI * 2.0)) * 180.0, 0.55 * 255, 255, 255), cv::COLOR_HLS2BGR));
        fillPaint(paint);
        fill();
    }

    beginPath();
    circle(cx, cy, r0 - 0.5f);
    circle(cx, cy, r1 + 0.5f);
    strokeColor(cv::Scalar(0, 0, 0, 64));
    strokeWidth(1.0f);
    stroke();

    // Selector
    save();
    translate(cx, cy);
    rotate((hue/255.0) * CV_PI * 2);

    // Marker on
    strokeWidth(2.0f);
    beginPath();
    rect(r0 - 1, -3, r1 - r0 + 2, 6);
    strokeColor(cv::Scalar(255, 255, 255, 192));
    stroke();

    paint = boxGradient(r0 - 3, -5, r1 - r0 + 6, 10, 2, 4, cv::Scalar(0, 0, 0, 128), cv::Scalar(0, 0, 0, 0));
    beginPath();
    rect(r0 - 2 - 10, -4 - 10, r1 - r0 + 4 + 20, 8 + 20);
    rect(r0 - 2, -4, r1 - r0 + 4, 8);
    pathWinding(NVG_HOLE);
    fillPaint(paint);
    fill();

    // Center triangle
    r = r0 - 6;
    ax = cosf(120.0f / 180.0f * NVG_PI) * r;
    ay = sinf(120.0f / 180.0f * NVG_PI) * r;
    bx = cosf(-120.0f / 180.0f * NVG_PI) * r;
    by = sinf(-120.0f / 180.0f * NVG_PI) * r;
    beginPath();
    moveTo(r, 0);
    lineTo(ax, ay);
    lineTo(bx, by);
    closePath();
    paint = linearGradient(r, 0, ax, ay, cv::v4d::colorConvert(cv::Scalar(hue, 128.0, 255.0, 255.0), cv::COLOR_HLS2BGR_FULL), cv::Scalar(255, 255, 255, 255));
    fillPaint(paint);
    fill();
    paint = linearGradient((r + ax) * 0.5f, (0 + ay) * 0.5f, bx, by, cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 255));
    fillPaint(paint);
    fill();
    strokeColor(cv::Scalar(0, 0, 0, 64));
    stroke();

    // Select circle on triangle
    ax = cosf(120.0f / 180.0f * NVG_PI) * r * 0.3f;
    ay = sinf(120.0f / 180.0f * NVG_PI) * r * 0.4f;
    strokeWidth(2.0f);
    beginPath();
    circle(ax, ay, 5);
    strokeColor(cv::Scalar(255, 255, 255, 192));
    stroke();

    paint = radialGradient(ax, ay, 7, 9, cv::Scalar(0, 0, 0, 64), cv::Scalar(0, 0, 0, 0));
    beginPath();
    rect(ax - 20, ay - 20, 40, 40);
    circle(ax, ay, 7);
    pathWinding(NVG_HOLE);
    fillPaint(paint);
    fill();

    restore();

    restore();
}

using namespace cv::v4d;

class NanoVGDemoPlan : public Plan {
	std::vector<cv::UMat> hsvChannels_;
	cv::UMat rgb_;
	cv::UMat bgra_;
	cv::UMat hsv_;
	cv::UMat hueChannel_;
	inline static long cnt_ = 0;
	double hue_ = 0;
public:
	using Plan::Plan;

	NanoVGDemoPlan(const cv::Rect& vp) : Plan(vp) {
		Global::registerShared(cnt_);
	}

	NanoVGDemoPlan(const cv::Size& sz) : NanoVGDemoPlan(cv::Rect(0, 0, sz.width, sz.height)) {
	}

	void infer(cv::Ptr<V4D> window) override {
		window->plain([](long& cnt, double& hue){
			long c;
			Global::lock(cnt);
			//we use frame count to calculate the current hue
			double t = ++c / 60.0;
			//nanovg hue fading depending on t
			hue = (sinf(t * 0.12) + 1.0) * 127.5;
			Global::unlock(cnt);
		},  cnt_, hue_);

		window->capture();

		//Acquire the framebuffer and convert it to RGB
		window->fb([](const cv::UMat &framebuffer, const cv::Rect& viewport, cv::UMat& rgb) {
			cvtColor(framebuffer(viewport), rgb, cv::COLOR_BGRA2RGB);
		}, viewport(), rgb_);

		window->plain([](cv::UMat& rgb, cv::UMat& hsv, std::vector<cv::UMat>& hsvChannels, double& hue){
			//Color-conversion from RGB to HSV
			cv::cvtColor(rgb, hsv, cv::COLOR_RGB2HSV_FULL);

			//Split the channels
			split(hsv,hsvChannels);
			//Set the current hue
			hsvChannels[0].setTo(std::round(hue));
			//Merge the channels back
			merge(hsvChannels,hsv);

			//Color-conversion from HSV to RGB
			cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB_FULL);
		}, rgb_, hsv_, hsvChannels_, hue_);

		//Acquire the framebuffer and convert the rgb_ into it
		window->fb([](cv::UMat &framebuffer, const cv::Rect& viewport, const cv::UMat& rgb) {
			cv::cvtColor(rgb, framebuffer(viewport), cv::COLOR_BGR2BGRA);
		}, viewport(), rgb_);

		//Render using nanovg
		window->nvg([](const cv::Size &sz, const double& h) {
			draw_color_wheel(sz.width - (sz.width / 5), sz.height - (sz.width / 5), sz.width / 6, sz.width / 6, h);
		}, size(), hue_);

		window->write();
	}
};

int main(int argc, char **argv) {
	if (argc != 2) {
        cerr << "Usage: nanovg-demo <video-file>" << endl;
        exit(1);
	}

    cv::Ptr<NanoVGDemoPlan> plan = new NanoVGDemoPlan(cv::Size(1280, 960));
    cv::Ptr<V4D> window = V4D::make(plan->size(), "NanoVG Demo", NANOVG);
    window->printSystemInfo();

    auto src = makeCaptureSource(window, argv[1]);
    auto sink = makeWriterSink(window, "nanovg-demo.mkv", src->fps(), plan->size());
    window->setSource(src);
    window->setSink(sink);

    window->run(plan);

    return 0;
}
