// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

constexpr unsigned int WIDTH = 1280;
constexpr unsigned int HEIGHT = 720;
constexpr bool OFFSCREEN = false;
#ifndef __EMSCRIPTEN__
constexpr const char *OUTPUT_FILENAME = "nanovg-demo.mkv";
#endif

using std::cerr;
using std::endl;

cv::Ptr<cv::v4d::V4D> window;

static void draw_color_wheel(float x, float y, float w, float h, float hue) {
    //color wheel drawing code taken from https://github.com/memononen/nanovg/blob/master/example/demo.c
    using namespace cv::v4d::nvg;
    clear({0,0,0,0});
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
    paint = linearGradient(r, 0, ax, ay, cv::v4d::colorConvert(cv::Scalar(hue, 128, 255, 255), cv::COLOR_HLS2BGR_FULL), cv::Scalar(255, 255, 255, 255));
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

static bool iteration() {
    static std::vector<cv::UMat> hsvChannels;
    static cv::UMat rgb;
    static cv::UMat bgra;
    static cv::UMat hsv;
    static cv::UMat hueChannel;

    //we use frame count to calculated the current hue
    float t = window->frameCount() / 60.0;
    //nanovg hue fading depending t
    float hue = (sinf(t * 0.12f) + 1.0f) * 127.5;

    if (!window->capture())
        return false;

    //Acquire the framebuffer and convert it to RGB
    window->fb([&](cv::UMat &frameBuffer) {
        cvtColor(frameBuffer, rgb, cv::COLOR_BGRA2RGB);
    });

    //Color-conversion from RGB to HSV
    cv::cvtColor(rgb, hsv, cv::COLOR_RGB2HSV_FULL);

    //Split the channels
    split(hsv,hsvChannels);
    //Set the current hue
    hsvChannels[0].setTo(hue);
    //Merge the channels back
    merge(hsvChannels,hsv);

    //Color-conversion from HSV to RGB
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB_FULL);

    //Acquire the framebuffer and convert the rgb into it
    window->fb([&](cv::UMat &frameBuffer) {
        cv::cvtColor(rgb, frameBuffer, cv::COLOR_RGB2BGRA);
    });

    //Render using nanovg
    window->nvg([&](const cv::Size &sz) {
    	//TODO automatically normalize hue between opencv and nanovg
        hue = ((170 + uint8_t(255 - hue))) % 255;
        draw_color_wheel(sz.width - 300, sz.height - 300, 250.0f, 250.0f, hue);
    });

    window->write();

    //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
    return window->display();
}

#ifndef __EMSCRIPTEN__
int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << "Usage: nanovg-demo <video-file>" << endl;
        exit(1);
    }
#else
int main() {
#endif
    using namespace cv::v4d;
    window = V4D::make(cv::Size(WIDTH, HEIGHT), cv::Size(), "NanoVG Demo", OFFSCREEN);
    window->printSystemInfo();

#ifndef __EMSCRIPTEN__
    Source src = makeCaptureSource(argv[1]);
    Sink sink = makeWriterSink(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), src.fps(), cv::Size(WIDTH, HEIGHT));
    window->setSource(src);
    window->setSink(sink);
#else
    Source src = makeCaptureSource(WIDTH, HEIGHT, window);
    window->setSource(src);
#endif

    window->run(iteration);

    return 0;
}
