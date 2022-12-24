#define CL_TARGET_OPENCL_VERSION 120

#include "../common/viz2d.hpp"
#include "../common/nvg.hpp"
#include "../common/util.hpp"

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
constexpr const int VA_HW_DEVICE_INDEX = 0;
constexpr double FPS = 60;
const cv::Scalar_<float> INITIAL_COLOR = kb::viz2d::color_convert(cv::Scalar(0.15 * 180.0, 128, 255, 255), cv::COLOR_HLS2BGR);
/** Visualization parameters **/


float min_star_size = 0.5f;
float max_star_size = 1.0f;
int min_star_count = 1000;
int max_star_count = 3000;
float star_alpha = 0.2;

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

cv::Ptr<kb::viz2d::Viz2D> v2d = new kb::viz2d::Viz2D(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Font Demo");
vector<string> lines;
bool update_stars = true;

void setup_gui(cv::Ptr<kb::viz2d::Viz2D> v2d) {
    v2d->makeWindow(5, 30, "Effect");
    v2d->makeGroup("Text Crawl");
    v2d->makeFormVariable("Font Size", font_size, 1.0f, 100.0f, true, "pt", "Font size of the text crawl");
    v2d->makeFormVariable("Warp Ratio", warp_ratio, 0.1f, 1.0f, true, "", "The ratio of start width to end width of a crawling line");

    v2d->makeColorPicker("Text Color", text_color, "The text color",[&](const nanogui::Color &c) {
        text_color[0] = c[0];
        text_color[1] = c[1];
        text_color[2] = c[2];
    });
    v2d->makeFormVariable("Alpha", text_alpha, 0.0f, 1.0f, true, "", "The opacity of the text");

    v2d->makeGroup("Stars");
    v2d->makeFormVariable("Min Star Size", min_star_size, 0.5f, 1.0f, true, "px", "Generate stars with this minimum size")
            ->set_callback([](const float& s){
        update_stars = true;
    });
    v2d->makeFormVariable("Max Star Size", max_star_size, 1.0f, 10.0f, true, "px", "Generate stars with this maximum size")
        ->set_callback([](const float& s){
                update_stars = true;
            });
    v2d->makeFormVariable("Min Star Count", min_star_count, 1, 1000, true, "", "Generate this minimum of stars")
        ->set_callback([](const float& s){
                update_stars = true;
            });
    v2d->makeFormVariable("Max Star Count", max_star_count, 1000, 5000, true, "", "Generate this maximum of stars")
        ->set_callback([](const float& s){
                update_stars = true;
            });
    v2d->makeFormVariable("Min Star Alpha", star_alpha, 0.2f, 1.0f, true, "", "Minimum opacity of stars")
        ->set_callback([](const float& s){
                update_stars = true;
            });

    v2d->makeWindow(8, 16, "Display");

    v2d->makeGroup("Display");
    v2d->makeFormVariable("Show FPS", show_fps, "Enable or disable the On-screen FPS display");
//    v2d->makeButton("Fullscreen", [=]() {
//        v2d->setFullscreen(!v2d->isFullscreen());
//    });
    v2d->makeButton("Offscreen", [=]() {
        v2d->setOffscreen(!v2d->isOffscreen());
    });
}

void iteration() {
    //BGRA
    static cv::UMat stars, warped;
    static cv::RNG rng(cv::getTickCount());
    static size_t cnt = 0;

    if(update_stars) {
        v2d->nanovg([&](const cv::Size& sz) {
            using namespace kb::viz2d::nvg;
            v2d->clear();
            //draw stars
            int numStars = rng.uniform(min_star_count, max_star_count);
            for(int i = 0; i < numStars; ++i) {
                beginPath();
                strokeWidth(rng.uniform(min_star_size, max_star_size));
                strokeColor(cv::Scalar(255, 255, 255, star_alpha * 255.0f));
                circle(rng.uniform(0, WIDTH) , rng.uniform(0, HEIGHT), 1);
                stroke();
            }
        });
        v2d->opencl([&](cv::UMat& frameBuffer){
            frameBuffer.copyTo(stars);
        });
        update_stars = false;
    }

    //Derive the transformation matrix tm for the pseudo 3D effect from quad1 and quad2.
    vector<cv::Point2f> quad1 = {{0,0},{WIDTH,0},{WIDTH,HEIGHT},{0,HEIGHT}};
    float l = std::round((WIDTH - (WIDTH * warp_ratio)) / 2.0);
    float r = WIDTH - l;

    vector<cv::Point2f> quad2 = {{l, 0.0f},{r, 0.0f},{WIDTH,HEIGHT},{0,HEIGHT}};
    cv::Mat tm = cv::getPerspectiveTransform(quad1, quad2);

    int y = 0;
    v2d->nanovg([&](const cv::Size& sz) {
        using namespace kb::viz2d::nvg;
        v2d->clear();

        fontSize(font_size);
        fontFace("sans-bold");
        fillColor(cv::Scalar(text_color.b() * 255.0f, text_color.g() * 255.0f, text_color.r() * 255.0f, text_alpha * 255.0f));
        textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);

        /** only draw lines that are visible **/

        //Total number of lines in the text
        off_t numLines = lines.size();
        //Height of the text in pixels
        off_t textHeight = (numLines * font_size);
        //How many pixels to translate the text up.
        off_t translateY = HEIGHT - cnt;
        translate(0, translateY);

        for (const auto &line : lines) {
            if (translateY + y > -textHeight && translateY + y <= HEIGHT) {
                text(WIDTH / 2.0, y, line.c_str(), line.c_str() + line.size());
                y += font_size;
            } else {
                //We can stop reading lines if the current line exceeds the page.
                break;
            }
        }
    });

    if(y == 0) {
        //Nothing drawn, exit.
//        exit(0);
    }

    v2d->opencl([&](cv::UMat& frameBuffer){
        //Pseudo 3D text effect.
        cv::warpPerspective(frameBuffer, warped, tm, frameBuffer.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        //Combine layers
        cv::add(stars, warped, frameBuffer);
    });

    update_fps(v2d, show_fps);

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
    using namespace kb::viz2d;

    print_system_info();
    if(!v2d->isOffscreen()) {
        setup_gui(v2d);
        v2d->setVisible(true);
    }
#ifndef __EMSCRIPTEN__
    v2d->makeVAWriter(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, v2d->getFrameBufferSize(), VA_HW_DEVICE_INDEX);
#endif

    //The text to display
    string txt = cv::getBuildInformation();
    //Save the text to a vector
    std::istringstream iss(txt);

    for (std::string line; std::getline(iss, line); ) {
        lines.push_back(line);
    }

    //Frame count.
#ifndef __EMSCRIPTEN__
    while(true)
        iteration();
#else
    emscripten_set_main_loop(iteration, -1, false);
#endif

    } catch(std::exception& ex) {
        cerr << "Exception: " << ex.what() << endl;
    }
    return 0;
}
