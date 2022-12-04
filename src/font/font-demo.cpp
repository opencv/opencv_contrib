#define CL_TARGET_OPENCL_VERSION 120

#include "../common/subsystems.hpp"

#include <string>
#include <algorithm>
#include <vector>
#include <sstream>
#include <limits>

/** Application parameters **/

constexpr unsigned int WIDTH = 1920;
constexpr unsigned int HEIGHT = 1080;
constexpr unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "font-demo.mkv";
constexpr const int VA_HW_DEVICE_INDEX = 0;
constexpr double FPS = 60;

/** Visualization parameters **/

constexpr float FONT_SIZE = 40.0f;
constexpr float MAX_STAR_SIZE = 1.0f;
constexpr int MIN_STAR_COUNT = 1000;
constexpr int MAX_STAR_COUNT = 3000;
constexpr float MIN_STAR_LIGHTNESS = 1.0f;
constexpr int MIN_STAR_ALPHA = 5;
// Intensity of glow defined by kernel size. The default scales with the image diagonal.
constexpr int GLOW_KERNEL_SIZE = std::max(int(DIAG / 138 % 2 == 0 ? DIAG / 138  + 1 : DIAG / 138), 1);

using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::istringstream;

int main(int argc, char **argv) {
    using namespace kb;

    //Initialize the application
    app::init("Font Demo", WIDTH, HEIGHT, OFFSCREEN);
    //Print system information
    app::print_system_info();

    //Initialize VP9 HW encoding using VAAPI
    cv::VideoWriter writer(OUTPUT_FILENAME, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, cv::Size(WIDTH, HEIGHT), {
            cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    //Copy OpenCL Context for VAAPI. Must be called right after first VideoWriter/VideoCapture initialization.
    va::copy();

    //BGRA
    cv::UMat frameBuffer, stars, warped;
    //BGR
    cv::UMat videoFrame;

    //The text to display
    string text = cv::getBuildInformation();
    //Save the text to a vector
    std::istringstream iss(text);
    vector<string> lines;
    for (std::string line; std::getline(iss, line); ) {
        lines.push_back(line);
    }

    //Derive the transformation matrix tm for the pseudo 3D effect from quad1 and quad2.
    vector<cv::Point2f> quad1 = {{0,0},{WIDTH,0},{WIDTH,HEIGHT},{0,HEIGHT}};
    vector<cv::Point2f> quad2 = {{WIDTH/3,0},{WIDTH/1.5,0},{WIDTH,HEIGHT},{0,HEIGHT}};
    cv::Mat tm = cv::getPerspectiveTransform(quad1, quad2);
    cv::RNG rng(cv::getTickCount());

    //Activate the OpenCL context for OpenGL.
    gl::bind();
    //Begin a nanovg frame.
    nvg::begin();
    nvg::clear(0,0,0,1);
    {
        //draw stars
        using kb::nvg::vg;
        int numStars = rng.uniform(MIN_STAR_COUNT, MAX_STAR_COUNT);
        for(int i = 0; i < numStars; ++i) {
            nvgBeginPath(vg);
            nvgStrokeWidth(vg, rng.uniform(0.5f, MAX_STAR_SIZE));
            nvgStrokeColor(vg, nvgHSLA(0, 1, rng.uniform(MIN_STAR_LIGHTNESS, 1.0f), rng.uniform(MIN_STAR_ALPHA, 255)));
            nvgCircle(vg, rng.uniform(0, WIDTH) , rng.uniform(0, HEIGHT), MAX_STAR_SIZE);
            nvgStroke(vg);
        }
    }
    //End a nanovg frame
    nvg::end();

    //Aquire frame buffer from OpenGL.
    gl::acquire_from_gl(frameBuffer);
    //Copy the star rendering.
    frameBuffer.copyTo(stars);
    //Release frame buffer to OpenGL.
    gl::release_to_gl(frameBuffer);

    //Frame count.
    size_t cnt = 0;
    //Y-position of the current line in pixels.
    float y;


    while (true) {
        y = 0;
        gl::bind();
        //Begin a nanovg frame.
        nvg::begin();
        //Clear the screen with black.
        nvg::clear();
        {
            using kb::nvg::vg;
            nvgBeginPath(vg);
            nvgFontSize(vg, FONT_SIZE);
            nvgFontFace(vg, "serif");
            nvgFillColor(vg, nvgHSLA(0.15, 1, 0.5, 255));
            nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);

            /** only draw lines that are visible **/

            //Total number of lines in the text
            off_t numLines = lines.size();
            //Height of the text in pixels
            off_t textHeight = (numLines * FONT_SIZE);
            //How many pixels to translate the text up.
            off_t translateY = HEIGHT - cnt;
            nvgTranslate(vg, 0, translateY);

            for (const auto &line : lines) {
                if (translateY + y > -textHeight && translateY + y <= HEIGHT) {
                    nvgText(vg, WIDTH / 2.0, y, line.c_str(), line.c_str() + line.size());
                    y += FONT_SIZE;
                } else {
                    //We can stop reading lines if the current line exceeds the page.
                    break;
                }
            }
        }
        //End a nanovg frame
        nvg::end();

        if(y == 0) {
            //Nothing drawn, exit.
            break;
        }

        //Aquire frame buffer from OpenGL.
        gl::acquire_from_gl(frameBuffer);
        //Pseudo 3D text effect.
        cv::warpPerspective(frameBuffer, warped, tm, videoFrame.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        //Combine layers
        cv::add(stars, warped, frameBuffer);
        //Color-conversion from BGRA to RGB. OpenCV/OpenCL.
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);
        //Transfer buffer ownership back to OpenGL.
        gl::release_to_gl(frameBuffer);

        //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
        if(!app::display())
            break;

        //Activate the OpenCL context for VAAPI.
        va::bind();
        //Encode the frame using VAAPI on the GPU.
        writer << videoFrame;

        ++cnt;
        //Wrap the cnt around if it becomes to big.
        if(cnt == std::numeric_limits<size_t>().max())
            cnt = 0;

        app::print_fps();
    }

    app::terminate();

    return 0;
}
