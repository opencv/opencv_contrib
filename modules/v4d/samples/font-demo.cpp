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
float min_star_size = 0.5f;
float max_star_size = 1.5f;
int min_star_count = 1000;
int max_star_count = 3000;
float star_alpha = 0.3f;

float font_size = 40.0f;
//nanogui::Color text_color = {INITIAL_COLOR[2] / 255.0f, INITIAL_COLOR[1] / 255.0f, INITIAL_COLOR[0] / 255.0f, INITIAL_COLOR[3] / 255.0f};
float text_alpha = 1.0;
float warp_ratio = 1.0f/3.0f;

bool show_fps = true;

using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::istringstream;

vector<string> lines;
bool update_stars = true;
bool update_perspective = true;

using namespace cv::v4d;
//static void setup_gui(cv::Ptr<V4D> window) {
//    window->nanogui([&](cv::v4d::FormHelper& form){
//        form.makeDialog(5, 30, "Effect");
//        form.makeGroup("Text Crawl");
//        form.makeFormVariable("Font Size", font_size, 1.0f, 100.0f, true, "pt", "Font size of the text crawl");
//        form.makeFormVariable("Warp Ratio", warp_ratio, 0.1f, 1.0f, true, "", "The ratio of start width to end width of a crawling line")->set_callback([&](const float &w) {
//            update_perspective = true;
//            warp_ratio = w;
//        });
//
//        form.makeColorPicker("Text Color", text_color, "The text color", [&](const nanogui::Color &c) {
//            text_color[0] = c[0];
//            text_color[1] = c[1];
//            text_color[2] = c[2];
//        });
//
//        form.makeFormVariable("Alpha", text_alpha, 0.0f, 1.0f, true, "", "The opacity of the text");
//
//        form.makeGroup("Stars");
//        form.makeFormVariable("Min Star Size", min_star_size, 0.5f, 1.0f, true, "px", "Generate stars with this minimum size")->set_callback([&](const float &s) {
//            update_stars = true;
//            min_star_size = s;
//        });
//        form.makeFormVariable("Max Star Size", max_star_size, 1.0f, 10.0f, true, "px", "Generate stars with this maximum size")->set_callback([&](const float &s) {
//            update_stars = true;
//            max_star_size = s;
//        });
//        form.makeFormVariable("Min Star Count", min_star_count, 1, 1000, true, "", "Generate this minimum of stars")->set_callback([&](const int &cnt) {
//            update_stars = true;
//            min_star_count = cnt;
//        });
//        form.makeFormVariable("Max Star Count", max_star_count, 1000, 5000, true, "", "Generate this maximum of stars")->set_callback([&](const int &cnt) {
//            update_stars = true;
//            max_star_count = cnt;
//        });
//        form.makeFormVariable("Min Star Alpha", star_alpha, 0.2f, 1.0f, true, "", "Minimum opacity of stars")->set_callback([&](const float &a) {
//            update_stars = true;
//            star_alpha = a;
//        });
//
//        form.makeDialog(8, 16, "Display");
//
//        form.makeGroup("Display");
//        form.makeFormVariable("Show FPS", show_fps, "Enable or disable the On-screen FPS display");
//    #ifndef __EMSCRIPTEN__
//        form.makeButton("Fullscreen", [=]() {
//            window->setFullscreen(!window->isFullscreen());
//        });
//    #endif
//        form.makeButton("Offscreen", [=]() {
//            window->setVisible(!window->isVisible());
//        });
//    });
//}

static bool iteration(cv::Ptr<V4D> window) {
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
        window->nvg([&](const cv::Size& sz) {
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
        });

        window->fb([&](cv::UMat& frameBuffer){
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

    window->nvg([&](const cv::Size& sz) {
        using namespace cv::v4d::nvg;
        clear();
        fontSize(font_size);
        fontFace("sans-bold");
        fillColor(cv::Scalar(100, 172, 255, 255));
        textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);

        /** only draw lines that are visible **/
        translate(0, translateY);

        for (size_t i = 0; i < lines.size(); ++i) {
            y = (i * font_size);
            if (y + translateY < textHeight && y + translateY + font_size > 0) {
                text(sz.width / 2.0, y, lines[i].c_str(), lines[i].c_str() + lines[i].size());
            }
        }
    });

    window->fb([&](cv::UMat& framebuffer) {
        //Pseudo 3D text effect.
        cv::warpPerspective(framebuffer, warped, tm, framebuffer.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        //Combine layers
        cv::add(stars, warped, framebuffer);
    });

    if(-translateY > textHeight) {
        //reset the scroll once the text is out of the picture
        cnt = 0;
    }

    window->write();

    ++cnt;
    //Wrap the cnt around if it becomes to big.
    if(cnt > std::numeric_limits<size_t>().max() / 2.0)
        cnt = 0;

    return window->display();
}

int main() {
    try {
        cv::Ptr<V4D> window = V4D::make(cv::Size(WIDTH, HEIGHT), cv::Size(), "Font Demo", OFFSCREEN);
//        if(!OFFSCREEN) {
//            setup_gui(window);
//        }

        window->printSystemInfo();

        //The text to display
        string txt = cv::getBuildInformation();
        //Save the text to a vector
        std::istringstream iss(txt);

        for (std::string line; std::getline(iss, line); ) {
            lines.push_back(line);
        }

#ifndef __EMSCRIPTEN__
        Sink sink = makeWriterSink(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, cv::Size(WIDTH, HEIGHT));
        window->setSink(sink);
#endif

        window->run(iteration);
    } catch(std::exception& ex) {
        cerr << "Exception: " << ex.what() << endl;
    }
    return 0;
}
