#define CL_TARGET_OPENCL_VERSION 120

#include "../common/viz2d.hpp"
#include "../common/util.hpp"
#include "../common/nvg.hpp"

constexpr unsigned int WIDTH = 1920;
constexpr unsigned int HEIGHT = 1080;
constexpr bool OFFSCREEN = false;
constexpr const char *OUTPUT_FILENAME = "nanovg-demo.mkv";
constexpr const int VA_HW_DEVICE_INDEX = 0;

using std::cerr;
using std::endl;

void drawColorwheel(float x, float y, float w, float h, float hue) {
    //color wheel drawing code taken from https://github.com/memononen/nanovg/blob/master/example/demo.c
    using namespace kb::viz2d;
    int i;
    float r0, r1, ax, ay, bx, by, cx, cy, aeps, r;
    nvg::Paint paint;

    nvg::save();

    cx = x + w * 0.5f;
    cy = y + h * 0.5f;
    r1 = (w < h ? w : h) * 0.5f - 5.0f;
    r0 = r1 - 20.0f;
    aeps = 0.5f / r1;   // half a pixel arc length in radians (2pi cancels out).

    for (i = 0; i < 6; i++) {
        float a0 = (float) i / 6.0f * CV_PI * 2.0f - aeps;
        float a1 = (float) (i + 1.0f) / 6.0f * CV_PI * 2.0f + aeps;
        nvg::beginPath();
        nvg::arc(cx, cy, r0, a0, a1, NVG_CW);
        nvg::arc(cx, cy, r1, a1, a0, NVG_CCW);
        nvg::closePath();
        ax = cx + cosf(a0) * (r0 + r1) * 0.5f;
        ay = cy + sinf(a0) * (r0 + r1) * 0.5f;
        bx = cx + cosf(a1) * (r0 + r1) * 0.5f;
        by = cy + sinf(a1) * (r0 + r1) * 0.5f;
        paint = nvg::linearGradient(ax, ay, bx, by,
                kb::viz2d::color_convert(cv::Scalar((a0 / (CV_PI * 2.0)) * 180.0, 0.55 * 255.0, 255.0, 255.0), cv::COLOR_HLS2BGR),
                kb::viz2d::color_convert(cv::Scalar((a1 / (CV_PI * 2.0)) * 180.0, 0.55 * 255, 255, 255), cv::COLOR_HLS2BGR));
        nvg::fillPaint(paint);
        nvg::fill();
    }

    nvg::beginPath();
    nvg::circle(cx, cy, r0 - 0.5f);
    nvg::circle(cx, cy, r1 + 0.5f);
    nvg::strokeColor(cv::Scalar(0, 0, 0, 64));
    nvg::strokeWidth(1.0f);
    nvg::stroke();

    // Selector
    nvg::save();
    nvg::translate(cx, cy);
    nvg::rotate((hue/255.0) * CV_PI * 2);

    // Marker on
    nvg::strokeWidth(2.0f);
    nvg::beginPath();
    nvg::rect(r0 - 1, -3, r1 - r0 + 2, 6);
    nvg::strokeColor(cv::Scalar(255, 255, 255, 192));
    nvg::stroke();

    paint = nvg::boxGradient(r0 - 3, -5, r1 - r0 + 6, 10, 2, 4, cv::Scalar(0, 0, 0, 128), cv::Scalar(0, 0, 0, 0));
    nvg::beginPath();
    nvg::rect(r0 - 2 - 10, -4 - 10, r1 - r0 + 4 + 20, 8 + 20);
    nvg::rect(r0 - 2, -4, r1 - r0 + 4, 8);
    nvg::pathWinding(NVG_HOLE);
    nvg::fillPaint(paint);
    nvg::fill();

    // Center triangle
    r = r0 - 6;
    ax = cosf(120.0f / 180.0f * NVG_PI) * r;
    ay = sinf(120.0f / 180.0f * NVG_PI) * r;
    bx = cosf(-120.0f / 180.0f * NVG_PI) * r;
    by = sinf(-120.0f / 180.0f * NVG_PI) * r;
    nvg::beginPath();
    nvg::moveTo(r, 0);
    nvg::lineTo(ax, ay);
    nvg::lineTo(bx, by);
    nvg::closePath();
    paint = nvg::linearGradient(r, 0, ax, ay, kb::viz2d::color_convert(cv::Scalar(hue, 128, 255, 255), cv::COLOR_HLS2BGR_FULL), cv::Scalar(255, 255, 255, 255));
    nvg::fillPaint(paint);
    nvg::fill();
    paint = nvg::linearGradient((r + ax) * 0.5f, (0 + ay) * 0.5f, bx, by, cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 255));
    nvg::fillPaint(paint);
    nvg::fill();
    nvg::strokeColor(cv::Scalar(0, 0, 0, 64));
    nvg::stroke();

    // Select circle on triangle
    ax = cosf(120.0f / 180.0f * NVG_PI) * r * 0.3f;
    ay = sinf(120.0f / 180.0f * NVG_PI) * r * 0.4f;
    nvg::strokeWidth(2.0f);
    nvg::beginPath();
    nvg::circle(ax, ay, 5);
    nvg::strokeColor(cv::Scalar(255, 255, 255, 192));
    nvg::stroke();

    paint = nvg::radialGradient(ax, ay, 7, 9, cv::Scalar(0, 0, 0, 64), cv::Scalar(0, 0, 0, 0));
    nvg::beginPath();
    nvg::rect(ax - 20, ay - 20, 40, 40);
    nvg::circle(ax, ay, 7);
    nvg::pathWinding(NVG_HOLE);
    nvg::fillPaint(paint);
    nvg::fill();

    nvg::restore();

    nvg::restore();
}

int main(int argc, char **argv) {
    using namespace kb::viz2d;
    if (argc != 2) {
        cerr << "Usage: nanovg-demo <video-file>" << endl;
        exit(1);
    }

    cv::Ptr<Viz2D> v2d = new Viz2D(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "NanoVG Demo");
    print_system_info();
    if (!v2d->isOffscreen())
        v2d->setVisible(true);

    auto capture = v2d->makeVACapture(argv[1], VA_HW_DEVICE_INDEX);
    if (!capture.isOpened()) {
        cerr << "ERROR! Unable to open video input" << endl;
        exit(-1);
    }

    float fps = capture.get(cv::CAP_PROP_FPS);
    float width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    float height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    v2d->makeVAWriter(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, v2d->getFrameBufferSize(), VA_HW_DEVICE_INDEX);

    cv::UMat rgb;
    cv::UMat bgra;
    cv::UMat hsv;
    cv::UMat hueChannel;

    while (true) {
        //we use time to calculated the current hue
        float time = cv::getTickCount() / cv::getTickFrequency();
        //nanovg hue fading between 0.0f and 255.0f
        float hue = (sinf(time * 0.12f) + 1.0f) * 127.5;

        if (!v2d->captureVA())
            break;

        v2d->opencl([&](cv::UMat &frameBuffer) {
            cvtColor(frameBuffer, rgb, cv::COLOR_BGRA2RGB);
            //Color-conversion from RGB to HSV. (OpenCL)
            cv::cvtColor(rgb, hsv, cv::COLOR_RGB2HSV_FULL);
            //Extract the hue channel
            cv::extractChannel(hsv, hueChannel, 0);
            //Set the current hue
            hueChannel.setTo(hue);
            //Insert the hue channel
            cv::insertChannel(hueChannel, hsv, 0);
            //Color-conversion from HSV to RGB. (OpenCL)
            cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB_FULL);
            //Color-conversion from RGB to BGRA. (OpenCL)
            cv::cvtColor(rgb, frameBuffer, cv::COLOR_RGB2BGRA);
        });

        //Render using nanovg
        v2d->nanovg([&](const cv::Size &sz) {
            hue = ((170 + uint8_t(255 - hue))) % 255;
            drawColorwheel(sz.width - 300, sz.height - 300, 250.0f, 250.0f, hue);
        });

        update_fps(v2d, true);

        v2d->writeVA();

        //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
        if(!v2d->display())
            break;
    }

    return 0;
}
