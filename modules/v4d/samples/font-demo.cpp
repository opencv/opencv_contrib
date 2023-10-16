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

using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::istringstream;

using namespace cv::v4d;

class FontDemoPlan : public Plan {
	struct Params {
		const cv::Scalar_<float> INITIAL_COLOR = cv::v4d::colorConvert(cv::Scalar(0.15 * 180.0, 128, 255, 255), cv::COLOR_HLS2RGB);
		float minStarSize_ = 0.5f;
		float maxStarSize_ = 1.5f;
		int minStarCount_ = 1000;
		int maxStarCount_ = 3000;
		float starAlpha_ = 0.3f;

		float fontSize_ = 40.0f;
		cv::Scalar_<float> textColor_ = INITIAL_COLOR / 255.0;
		float warpRatio_ = 1.0f/3.0f;
	} params_;

	vector<string> lines_;
	bool updateStars_ = true;
	bool updatePerspective_ = true;

    //BGRA
    cv::UMat stars_, warped_, frame_;
    //transformation matrix
    cv::Mat tm_;
    cv::RNG rng_ = cv::getTickCount();
    //line count
    uint32_t cnt_ = 0;
    //Total number of lines in the text
    int32_t numLines_;
    //Height of the text in pixels
    int32_t textHeight_;
    //y-value of the current line
    int32_t y_ = 0;

    int32_t translateY_;
public:
    void gui(cv::Ptr<V4D> window) override {
        window->imgui([this](cv::Ptr<V4D> window, ImGuiContext* ctx){
            using namespace ImGui;
            SetCurrentContext(ctx);
            Begin("Effect");
            Text("Text Crawl");
            SliderFloat("Font Size", &params_.fontSize_, 1.0f, 100.0f);
            if(SliderFloat("Warp Ratio", &params_.warpRatio_, 0.1f, 1.0f))
                updatePerspective_ = true;
            ColorPicker4("Text Color", params_.textColor_.val);
            Text("Stars");

            if(SliderFloat("Min Star Size", &params_.minStarSize_, 0.5f, 1.0f))
                updateStars_ = true;
            if(SliderFloat("Max Star Size", &params_.maxStarSize_, 1.0f, 10.0f))
                updateStars_ = true;
            if(SliderInt("Min Star Count", &params_.minStarCount_, 1, 1000))
                updateStars_ = true;
            if(SliderInt("Max Star Count", &params_.maxStarCount_, 1000, 5000))
                updateStars_ = true;
            if(SliderFloat("Min Star Alpha", &params_.starAlpha_, 0.2f, 1.0f))
                updateStars_ = true;
            End();
        });
    }

    void setup(cv::Ptr<V4D> window) override {
        //The text to display
        string txt = cv::getBuildInformation();
        //Save the text to a vector
        std::istringstream iss(txt);

        for (std::string line; std::getline(iss, line); ) {
            lines_.push_back(line);
        }
        numLines_ = lines_.size();
        textHeight_ = (numLines_ * params_.fontSize_);
    }

    void infer(cv::Ptr<V4D> window) override {
    	auto always = []() { return true; };
    	auto isTrue = [](const bool& b) { return b; };

		window->graph(isTrue, updateStars_);
		{
			window->nvg([](const cv::Size& sz, cv::RNG& rng, const Params& params) {
				using namespace cv::v4d::nvg;
				clear();

				//draw stars
				int numStars = rng.uniform(params.minStarCount_, params.maxStarCount_);
				for(int i = 0; i < numStars; ++i) {
					beginPath();
					const auto& size = rng.uniform(params.minStarSize_, params.maxStarSize_);
					strokeWidth(size);
					strokeColor(cv::Scalar(255, 255, 255, params.starAlpha_ * 255.0f));
					circle(rng.uniform(0, sz.width) , rng.uniform(0, sz.height), size / 2.0);
					stroke();
				}
			}, window->fbSize(), rng_, params_);

			window->fb([](const cv::UMat& frameBuffer, cv::UMat& f){
				frameBuffer.copyTo(f);
			}, stars_);
		}
		window->endgraph(isTrue, updateStars_);

		window->graph(isTrue, updatePerspective_);
		{
			window->parallel([](cv::Mat& tm, const Params& params){
				//Derive the transformation matrix tm for the pseudo 3D effect from quad1 and quad2.
				vector<cv::Point2f> quad1 = {{0,0},{WIDTH,0},{WIDTH,HEIGHT},{0,HEIGHT}};
				float l = (WIDTH - (WIDTH * params.warpRatio_)) / 2.0;
				float r = WIDTH - l;

				vector<cv::Point2f> quad2 = {{l, 0.0f},{r, 0.0f},{WIDTH,HEIGHT},{0,HEIGHT}};
				tm = cv::getPerspectiveTransform(quad1, quad2);
			}, tm_, params_);
		}
		window->endgraph(isTrue, updatePerspective_);

		window->graph(always);
		{
			window->nvg([](const cv::Size& sz, int32_t& ty, const int32_t& cnt, int32_t& y, const int32_t& textHeight, const std::vector<std::string> lines, const Params& params) {
				//How many pixels to translate the text up.
		    	ty = HEIGHT - cnt;
				using namespace cv::v4d::nvg;
				clear();
				fontSize(params.fontSize_);
				fontFace("sans-bold");
				fillColor(params.textColor_ * 255);
				textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);

				/** only draw lines that are visible **/
				translate(0, ty);

				for (size_t i = 0; i < lines.size(); ++i) {
					y = (i * params.fontSize_);
					if (y + ty < textHeight && y + ty + params.fontSize_ > 0) {
						text(sz.width / 2.0, y, lines[i].c_str(), lines[i].c_str() + lines[i].size());
					}
				}
			}, window->fbSize(), translateY_, cnt_, y_, textHeight_, lines_, params_);

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

			window->write([](cv::UMat& outputFrame, const cv::UMat& f, bool& updatePerspective, bool& updateStars){
				f.copyTo(outputFrame);
				updatePerspective = false;
				updateStars = false;
			}, frame_, updatePerspective_, updateStars_);
		}
		window->endgraph(always);

    }
};
int main() {
    try {
        cv::Ptr<V4D> window = V4D::make(WIDTH, HEIGHT, "Font Demo", ALL, OFFSCREEN);

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
