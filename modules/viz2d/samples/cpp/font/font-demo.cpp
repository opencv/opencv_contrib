// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/viz2d/viz2d.hpp"
#include "opencv2/viz2d/nvg.hpp"
#include "opencv2/viz2d/util.hpp"

#include <string>
#include <algorithm>
#include <vector>
#include <sstream>
#include <limits>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

/** Application parameters **/
constexpr unsigned int WIDTH = 1920;
constexpr unsigned int HEIGHT = 1080;
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "font-demo.mkv";
constexpr double FPS = 60;
const cv::Scalar_<float> INITIAL_COLOR = cv::viz::colorConvert(cv::Scalar(0.15 * 180.0, 128, 255, 255), cv::COLOR_HLS2BGR);

/** Visualization parameters **/
float min_star_size = 0.5f;
float max_star_size = 1.0f;
int min_star_count = 1000;
int max_star_count = 3000;
float star_alpha = 0.3;

float font_size = 40.0f;
nanogui::Color text_color = {INITIAL_COLOR[2] / 255.0f, INITIAL_COLOR[1] / 255.0f, INITIAL_COLOR[0] / 255.0f, INITIAL_COLOR[3] / 255.0f};
float text_alpha = 1.0;
float warp_ratio = 1.0f/3.0f;

bool show_fps = true;

using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::istringstream;

cv::Ptr<cv::viz::Viz2D> v2d = cv::viz::Viz2D::make(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Font Demo");
vector<string> lines;
bool update_stars = true;
bool update_perspective = true;

void setup_gui(cv::Ptr<cv::viz::Viz2D> v2d) {
    v2d->nanogui([&](cv::viz::FormHelper& form){
        form.makeDialog(5, 30, "Effect");
        form.makeGroup("Text Crawl");
        form.makeFormVariable("Font Size", font_size, 1.0f, 100.0f, true, "pt", "Font size of the text crawl");
        form.makeFormVariable("Warp Ratio", warp_ratio, 0.1f, 1.0f, true, "", "The ratio of start width to end width of a crawling line")->set_callback([&](const float &w) {
            update_perspective = true;
            warp_ratio = w;
        });

        form.makeColorPicker("Text Color", text_color, "The text color", [&](const nanogui::Color &c) {
            text_color[0] = c[0];
            text_color[1] = c[1];
            text_color[2] = c[2];
        });

        form.makeFormVariable("Alpha", text_alpha, 0.0f, 1.0f, true, "", "The opacity of the text");

        form.makeGroup("Stars");
        form.makeFormVariable("Min Star Size", min_star_size, 0.5f, 1.0f, true, "px", "Generate stars with this minimum size")->set_callback([](const float &s) {
            update_stars = true;
            min_star_size = s;
        });
        form.makeFormVariable("Max Star Size", max_star_size, 1.0f, 10.0f, true, "px", "Generate stars with this maximum size")->set_callback([](const float &s) {
            update_stars = true;
            max_star_size = s;
        });
        form.makeFormVariable("Min Star Count", min_star_count, 1, 1000, true, "", "Generate this minimum of stars")->set_callback([](const int &cnt) {
            update_stars = true;
            min_star_count = cnt;
        });
        form.makeFormVariable("Max Star Count", max_star_count, 1000, 5000, true, "", "Generate this maximum of stars")->set_callback([](const int &cnt) {
            update_stars = true;
            max_star_count = cnt;
        });
        form.makeFormVariable("Min Star Alpha", star_alpha, 0.2f, 1.0f, true, "", "Minimum opacity of stars")->set_callback([](const float &a) {
            update_stars = true;
            star_alpha = a;
        });

        form.makeDialog(8, 16, "Display");

        form.makeGroup("Display");
        form.makeFormVariable("Show FPS", show_fps, "Enable or disable the On-screen FPS display");
    #ifndef __EMSCRIPTEN__
        form.makeButton("Fullscreen", [=]() {
            v2d->setFullscreen(!v2d->isFullscreen());
        });
    #endif
        form.makeButton("Offscreen", [=]() {
            v2d->setOffscreen(!v2d->isOffscreen());
        });
    });
}

void iteration() {
    //BGRA
    static cv::UMat stars, warped;
    //transformation matrix
    static cv::Mat tm;
    static cv::RNG rng(cv::getTickCount());
    //line count
    static uint32_t cnt = 0;
    //Total number of lines in the text
    static int32_t numLines = lines.size();
    //Height of the text in pixels
    static int32_t textHeight = (numLines * font_size);
    //y-value of the current line
    static int32_t y = 0;
    //How many pixels to translate the text up.
    int32_t translateY = HEIGHT - cnt;

    if(update_stars) {
        v2d->nvg([&](const cv::Size& sz) {
            using namespace cv::viz::nvg;
            v2d->clear();
            //draw stars
            int numStars = rng.uniform(min_star_count, max_star_count);
            for(int i = 0; i < numStars; ++i) {
                beginPath();
                const auto& size = rng.uniform(min_star_size, max_star_size);
                strokeWidth(size);
                strokeColor(cv::Scalar(255, 255, 255, star_alpha * 255.0f));
                circle(rng.uniform(0, WIDTH) , rng.uniform(0, HEIGHT), size / 2.0);
                stroke();
            }
        });
        v2d->fb([&](cv::UMat& frameBuffer){
            frameBuffer.copyTo(stars);
        });
        update_stars = false;
    }

    if(update_perspective) {
        //Derive the transformation matrix tm for the pseudo 3D effect from quad1 and quad2.
        vector<cv::Point2f> quad1 = {{0,0},{WIDTH,0},{WIDTH,HEIGHT},{0,HEIGHT}};
        float l = (WIDTH - (WIDTH * warp_ratio)) / 2.0;
        float r = WIDTH - l;

        vector<cv::Point2f> quad2 = {{l, 0.0f},{r, 0.0f},{WIDTH,HEIGHT},{0,HEIGHT}};
        tm = cv::getPerspectiveTransform(quad1, quad2);
        update_perspective = false;
    }

    v2d->nvg([&](const cv::Size& sz) {
        using namespace cv::viz::nvg;
        v2d->clear();

        fontSize(font_size);
        fontFace("sans-bold");
        fillColor(cv::Scalar(text_color.b() * 255.0f, text_color.g() * 255.0f, text_color.r() * 255.0f, text_alpha * 255.0f));
        textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);

        /** only draw lines that are visible **/
        translate(0, translateY);

        for (size_t i = 0; i < lines.size(); ++i) {
            y = (i * font_size);
            if (y + translateY < textHeight && y + translateY + font_size > 0) {
                text(WIDTH / 2.0, y, lines[i].c_str(), lines[i].c_str() + lines[i].size());
            }
        }
    });

    v2d->fb([&](cv::UMat& frameBuffer){
        //Pseudo 3D text effect.
        cv::warpPerspective(frameBuffer, warped, tm, frameBuffer.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        //Combine layers
        cv::add(stars, warped, frameBuffer);
    });

    if(-translateY > textHeight) {
        //reset the scroll once the text is out of the picture
        cnt = 0;
    }

    updateFps(v2d, show_fps);

#ifndef __EMSCRIPTEN__
    v2d->write();
#endif

    //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
    if(!v2d->display())
        exit(0);

    ++cnt;
    //Wrap the cnt around if it becomes to big.
    if(cnt > std::numeric_limits<size_t>().max() / 2.0)
        cnt = 0;
}

int main(int argc, char **argv) {
    try {
        using namespace cv::viz;

        printSystemInfo();
        if(!v2d->isOffscreen()) {
            setup_gui(v2d);
            v2d->setVisible(true);
        }

        //The text to display
        string txt = cv::getBuildInformation();
        //Save the text to a vector
        std::istringstream iss(txt);

        for (std::string line; std::getline(iss, line); ) {
            lines.push_back(line);
        }

    #ifndef __EMSCRIPTEN__
        Sink sink = makeWriterSink(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, cv::Size(WIDTH, HEIGHT));
        v2d->setSink(sink);
    #endif

        v2d->run(iteration);
    } catch(std::exception& ex) {
        cerr << "Exception: " << ex.what() << endl;
    }
    return 0;
}
