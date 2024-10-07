// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

using namespace cv::v4d;


class DisplayImageBgfx : public Plan {
	Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);
public:
	void setup() override {
		bgfx([](const cv::Rect& vp) {
			// Set view 0 clear state.
			bgfx::setViewClear(0
				, BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH
				, 0x303030ff
				, 1.0f
				, 0
				);

			// Set view 0 default viewport.
			bgfx::setViewRect(0, vp.x, vp.y, uint16_t(vp.width), uint16_t(vp.height));
		}, vp_);
	}

	void infer() override {
		bgfx([](const cv::Rect& vp) {

			// This dummy draw call is here to make sure that view 0 is cleared
			// if no other draw calls are submitted to view 0.
			bgfx::touch(0);

			// Use debug font to print information about this example.
			bgfx::dbgTextClear();

			const bgfx::Stats* stats = bgfx::getStats();

			bgfx::dbgTextPrintf(
				  bx::max<uint16_t>(uint16_t(stats->textWidth/2), 20)-20
				, bx::max<uint16_t>(uint16_t(stats->textHeight/2),  6)-6
				, 40
				, "Hello %s"
				, "World"
				);
			bgfx::dbgTextPrintf(0, 1, 0x0f, "Color can be changed with ANSI \x1b[9;me\x1b[10;ms\x1b[11;mc\x1b[12;ma\x1b[13;mp\x1b[14;me\x1b[0m code too.");

			bgfx::dbgTextPrintf(80, 1, 0x0f, "\x1b[;0m    \x1b[;1m    \x1b[; 2m    \x1b[; 3m    \x1b[; 4m    \x1b[; 5m    \x1b[; 6m    \x1b[; 7m    \x1b[0m");
			bgfx::dbgTextPrintf(80, 2, 0x0f, "\x1b[;8m    \x1b[;9m    \x1b[;10m    \x1b[;11m    \x1b[;12m    \x1b[;13m    \x1b[;14m    \x1b[;15m    \x1b[0m");

			bgfx::dbgTextPrintf(0, 2, 0x0f, "Backbuffer %dW x %dH in pixels, debug text %dW x %dH in characters."
				, stats->width
				, stats->height
				, stats->textWidth
				, stats->textHeight
				);

			// Advance to next frame. Rendering thread will be kicked to
			// process submitted rendering primitives.
			bgfx::frame();

		}, vp_);
	}
};


int main(int argc, char** argv) {
	cv::Rect viewport(0,0, 1280, 720);
	cv::Ptr<V4D> runtime = V4D::init(viewport, "Display an image using bgfx", AllocateFlags::BGFX | AllocateFlags::IMGUI);
    Plan::run<DisplayImageBgfx>(0);

    return 0;
}
