// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

#include <string>
#include <algorithm>
#include <vector>
#include <sstream>
#include <limits>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

/* Demo parameters */
#ifndef __EMSCRIPTEN__
constexpr long unsigned int WIDTH = 1280;
constexpr long unsigned int HEIGHT = 720;
#else
constexpr long unsigned int WIDTH = 960;
constexpr long unsigned int HEIGHT = 960;
#endif
constexpr bool OFFSCREEN = false;
#ifndef __EMSCRIPTEN__
constexpr const char* OUTPUT_FILENAME = "font-demo.mkv";
constexpr double FPS = 60;
#endif
const cv::Scalar_<float> INITIAL_COLOR = cv::v4d::colorConvert(cv::Scalar(0.15 * 180.0, 128, 255, 255), cv::COLOR_HLS2BGR);

/* Visualization parameters */
static float min_star_size = 0.5f;
static float max_star_size = 1.5f;
static int min_star_count = 1000;
static int max_star_count = 3000;
static float star_alpha = 0.3f;

static float font_size = 40.0f;
static float text_color[4] = {INITIAL_COLOR[2] / 255.0f, INITIAL_COLOR[1] / 255.0f, INITIAL_COLOR[0] / 255.0f, INITIAL_COLOR[3] / 255.0f};
static float warp_ratio = 1.0f/3.0f;

using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::istringstream;

static thread_local vector<string> lines;
static thread_local bool update_stars = true;
static thread_local bool update_perspective = true;

using namespace cv::v4d;
static void setup_gui(cv::Ptr<V4D> window) {
    window->imgui([](ImGuiContext* ctx){
        using namespace ImGui;
        SetCurrentContext(ctx);
        Begin("Effect");
        Text("Text Crawl");
        SliderFloat("Font Size", &font_size, 1.0f, 100.0f);
        if(SliderFloat("Warp Ratio", &warp_ratio, 0.1f, 1.0f))
            update_perspective = true;
        ColorPicker4("Text Color", text_color);
        Text("Stars");

        if(SliderFloat("Min Star Size", &min_star_size, 0.5f, 1.0f))
            update_stars = true;
        if(SliderFloat("Max Star Size", &max_star_size, 1.0f, 10.0f))
            update_stars = true;
        if(SliderInt("Min Star Count", &min_star_count, 1, 1000))
            update_stars = true;
        if(SliderInt("Max Star Count", &max_star_count, 1000, 5000))
            update_stars = true;
        if(SliderFloat("Min Star Alpha", &star_alpha, 0.2f, 1.0f))
            update_stars = true;
        End();
    });
}

class FontDemoPlan : public Plan {
    //BGRA
    cv::UMat stars_, warped_, frame_;
    //transformation matrix
    cv::Mat tm_;
    cv::RNG rng_ = cv::getTickCount();
    //line count
    uint32_t cnt_ = 0;
    //Total number of lines in the text
    int32_t numLines_ = lines.size();
    //Height of the text in pixels
    int32_t textHeight_ = (numLines_ * font_size);
    //y-value of the current line
    int32_t y_ = 0;

    int32_t translateY_;
public:
    void infer(cv::Ptr<V4D> window) override {
    	auto always = []() { return true; };
    	auto isTrue = [](const bool& b) { return b; };

		window->graph(isTrue, update_stars);
		{
			window->nvg([](const cv::Size& sz, cv::RNG& rng) {
				using namespace cv::v4d::nvg;
				clear();

				//draw stars
				int numStars = rng.uniform(min_star_count, max_star_count);
				for(int i = 0; i < numStars; ++i) {
					beginPath();
					const auto& size = rng.uniform(min_star_size, max_star_size);
					strokeWidth(size);
					strokeColor(cv::Scalar(255, 255, 255, star_alpha * 255.0f));
					circle(rng.uniform(0, sz.width) , rng.uniform(0, sz.height), size / 2.0);
					stroke();
				}
			}, window->fbSize(), rng_);

			window->fb([](const cv::UMat& frameBuffer, cv::UMat& f){
				frameBuffer.copyTo(f);
			}, stars_);
		}
		window->endgraph(isTrue, update_stars);

		window->graph(isTrue, update_perspective);
		{
			window->parallel([](cv::Mat& tm){
				//Derive the transformation matrix tm for the pseudo 3D effect from quad1 and quad2.
				vector<cv::Point2f> quad1 = {{0,0},{WIDTH,0},{WIDTH,HEIGHT},{0,HEIGHT}};
				float l = (WIDTH - (WIDTH * warp_ratio)) / 2.0;
				float r = WIDTH - l;

				vector<cv::Point2f> quad2 = {{l, 0.0f},{r, 0.0f},{WIDTH,HEIGHT},{0,HEIGHT}};
				tm = cv::getPerspectiveTransform(quad1, quad2);
			}, tm_);
		}
		window->endgraph(isTrue, update_perspective);

		window->graph(always);
		{
			window->nvg([](const cv::Size& sz, int32_t& ty, const int32_t& cnt, int32_t& y, const int32_t& textHeight) {
				//How many pixels to translate the text up.
		    	ty = HEIGHT - cnt;
				using namespace cv::v4d::nvg;
				clear();
				fontSize(font_size);
				fontFace("sans-bold");
				fillColor(cv::Scalar(text_color[2] * 255.0f, text_color[1] * 255.0f, text_color[0] * 255.0f, text_color[3] * 255.0f));
				textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);

				/** only draw lines that are visible **/
				translate(0, ty);

				for (size_t i = 0; i < lines.size(); ++i) {
					y = (i * font_size);
					if (y + ty < textHeight && y + ty + font_size > 0) {
						text(sz.width / 2.0, y, lines[i].c_str(), lines[i].c_str() + lines[i].size());
					}
				}
			}, window->fbSize(), translateY_, cnt_, y_, textHeight_);

			window->fb([](cv::UMat& framebuffer, cv::UMat& w, cv::UMat& s, cv::Mat& t, cv::UMat& f) {
				//Pseudo 3D text effect.
				cv::warpPerspective(framebuffer, w, t, framebuffer.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
				//Combine layers
				cv::add(s, w, framebuffer);
				framebuffer.copyTo(f);
			}, warped_, stars_, tm_, frame_);

			window->parallel([](const int32_t& translateY, const int32_t& textHeight, uint32_t& cnt) {
				if(-translateY > textHeight) {
					//reset the scroll once the text is out of the picture
					cnt = 0;
				}

				++cnt;
				//Wrap the cnt around if it becomes to big.
				if(cnt > std::numeric_limits<uint32_t>().max() / 2.0)
					cnt = 0;
			}, translateY_, textHeight_, cnt_);

			window->write([](cv::UMat& outputFrame, cv::UMat& f){
				f.copyTo(outputFrame);
				update_perspective = false;
				update_stars = false;
			}, frame_);
		}
		window->endgraph(always);

    }
};
int main() {
    try {
        cv::Ptr<V4D> window = V4D::make(WIDTH, HEIGHT, "Font Demo", ALL, OFFSCREEN);

        if(!OFFSCREEN) {
            setup_gui(window);
        }

        window->printSystemInfo();

        //The text to display
        string txt = cv::getBuildInformation();
        //Save the text to a vector
        std::istringstream iss(txt);

        for (std::string line; std::getline(iss, line); ) {
            lines.push_back(line);
        }

#ifndef __EMSCRIPTEN__
        auto sink = makeWriterSink(window, OUTPUT_FILENAME, FPS, cv::Size(WIDTH, HEIGHT));
        window->setSink(sink);
#endif

        window->run<FontDemoPlan>(0);
    } catch(std::exception& ex) {
        cerr << "Exception: " << ex.what() << endl;
    }
    return 0;
}
