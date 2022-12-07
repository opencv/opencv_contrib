#define CL_TARGET_OPENCL_VERSION 120

#include "../common/subsystems.hpp"

constexpr unsigned int WIDTH = 1920;
constexpr unsigned int HEIGHT = 1080;
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "nanovg-demo.mkv";
constexpr const int VA_HW_DEVICE_INDEX = 0;

using std::cerr;
using std::endl;

void drawColorwheel(NVGcontext *vg, float x, float y, float w, float h, float hue) {
    //color wheel drawing code taken from https://github.com/memononen/nanovg/blob/master/example/demo.c
    int i;
    float r0, r1, ax, ay, bx, by, cx, cy, aeps, r;
    NVGpaint paint;

    nvgSave(vg);

    /*  nvgBeginPath(vg);
     nvgRect(vg, x,y,w,h);
     nvgFillColor(vg, nvgRGBA(255,0,0,128));
     nvgFill(vg);*/

    cx = x + w * 0.5f;
    cy = y + h * 0.5f;
    r1 = (w < h ? w : h) * 0.5f - 5.0f;
    r0 = r1 - 20.0f;
    aeps = 0.5f / r1;   // half a pixel arc length in radians (2pi cancels out).

    for (i = 0; i < 6; i++) {
        float a0 = (float) i / 6.0f * NVG_PI * 2.0f - aeps;
        float a1 = (float) (i + 1.0f) / 6.0f * NVG_PI * 2.0f + aeps;
        nvgBeginPath(vg);
        nvgArc(vg, cx, cy, r0, a0, a1, NVG_CW);
        nvgArc(vg, cx, cy, r1, a1, a0, NVG_CCW);
        nvgClosePath(vg);
        ax = cx + cosf(a0) * (r0 + r1) * 0.5f;
        ay = cy + sinf(a0) * (r0 + r1) * 0.5f;
        bx = cx + cosf(a1) * (r0 + r1) * 0.5f;
        by = cy + sinf(a1) * (r0 + r1) * 0.5f;
        paint = nvgLinearGradient(vg, ax, ay, bx, by, nvgHSLA(a0 / (NVG_PI * 2), 1.0f, 0.55f, 255), nvgHSLA(a1 / (NVG_PI * 2), 1.0f, 0.55f, 255));
        nvgFillPaint(vg, paint);
        nvgFill(vg);
    }

    nvgBeginPath(vg);
    nvgCircle(vg, cx, cy, r0 - 0.5f);
    nvgCircle(vg, cx, cy, r1 + 0.5f);
    nvgStrokeColor(vg, nvgRGBA(0, 0, 0, 64));
    nvgStrokeWidth(vg, 1.0f);
    nvgStroke(vg);

    // Selector
    nvgSave(vg);
    nvgTranslate(vg, cx, cy);
    nvgRotate(vg, hue * NVG_PI * 2);

    // Marker on
    nvgStrokeWidth(vg, 2.0f);
    nvgBeginPath(vg);
    nvgRect(vg, r0 - 1, -3, r1 - r0 + 2, 6);
    nvgStrokeColor(vg, nvgRGBA(255, 255, 255, 192));
    nvgStroke(vg);

    paint = nvgBoxGradient(vg, r0 - 3, -5, r1 - r0 + 6, 10, 2, 4, nvgRGBA(0, 0, 0, 128), nvgRGBA(0, 0, 0, 0));
    nvgBeginPath(vg);
    nvgRect(vg, r0 - 2 - 10, -4 - 10, r1 - r0 + 4 + 20, 8 + 20);
    nvgRect(vg, r0 - 2, -4, r1 - r0 + 4, 8);
    nvgPathWinding(vg, NVG_HOLE);
    nvgFillPaint(vg, paint);
    nvgFill(vg);

    // Center triangle
    r = r0 - 6;
    ax = cosf(120.0f / 180.0f * NVG_PI) * r;
    ay = sinf(120.0f / 180.0f * NVG_PI) * r;
    bx = cosf(-120.0f / 180.0f * NVG_PI) * r;
    by = sinf(-120.0f / 180.0f * NVG_PI) * r;
    nvgBeginPath(vg);
    nvgMoveTo(vg, r, 0);
    nvgLineTo(vg, ax, ay);
    nvgLineTo(vg, bx, by);
    nvgClosePath(vg);
    paint = nvgLinearGradient(vg, r, 0, ax, ay, nvgHSLA(hue, 1.0f, 0.5f, 255), nvgRGBA(255, 255, 255, 255));
    nvgFillPaint(vg, paint);
    nvgFill(vg);
    paint = nvgLinearGradient(vg, (r + ax) * 0.5f, (0 + ay) * 0.5f, bx, by, nvgRGBA(0, 0, 0, 0), nvgRGBA(0, 0, 0, 255));
    nvgFillPaint(vg, paint);
    nvgFill(vg);
    nvgStrokeColor(vg, nvgRGBA(0, 0, 0, 64));
    nvgStroke(vg);

    // Select circle on triangle
    ax = cosf(120.0f / 180.0f * NVG_PI) * r * 0.3f;
    ay = sinf(120.0f / 180.0f * NVG_PI) * r * 0.4f;
    nvgStrokeWidth(vg, 2.0f);
    nvgBeginPath(vg);
    nvgCircle(vg, ax, ay, 5);
    nvgStrokeColor(vg, nvgRGBA(255, 255, 255, 192));
    nvgStroke(vg);

    paint = nvgRadialGradient(vg, ax, ay, 7, 9, nvgRGBA(0, 0, 0, 64), nvgRGBA(0, 0, 0, 0));
    nvgBeginPath(vg);
    nvgRect(vg, ax - 20, ay - 20, 40, 40);
    nvgCircle(vg, ax, ay, 7);
    nvgPathWinding(vg, NVG_HOLE);
    nvgFillPaint(vg, paint);
    nvgFill(vg);

    nvgRestore(vg);

    nvgRestore(vg);
}

int main(int argc, char **argv) {
    using namespace kb;
    if (argc != 2) {
        cerr << "Usage: nanovg-demo <video-file>" << endl;
        exit(1);
    }

    //Initialize the application
    app::init("Nanovg Demo", WIDTH, HEIGHT, OFFSCREEN);
    //Print system information
    app::print_system_info();

    app::run([&]() {
        //Initialize MJPEG HW decoding using VAAPI
        cv::VideoCapture capture(argv[1], cv::CAP_FFMPEG, {
                cv::CAP_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
                cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
                cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
        });

        //Copy OpenCL Context for VAAPI. Must be called right after first VideoWriter/VideoCapture initialization.
        va::copy();

        // Check if we succeeded
        if (!capture.isOpened()) {
            cerr << "ERROR! Unable to open video input" << endl;
            return;
        }

        double fps = capture.get(cv::CAP_PROP_FPS);

        //Initialize VP9 HW encoding using VAAPI
        cv::VideoWriter writer(OUTPUT_FILENAME, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, cv::Size(WIDTH, HEIGHT), {
                cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
                cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
        });

        cv::UMat rgb;
        cv::UMat bgra;
        cv::UMat hsv;
        cv::UMat hueChannel;

        while (true) {
            //we use time to calculated the current hue
            float time = cv::getTickCount() / cv::getTickFrequency();
            //nanovg hue fading between 0.0f and 1.0f
            float nvgHue = (sinf(time*0.12f)+1.0f) / 2.0f;
            //opencv hue fading between 0 and 255
            int cvHue = (42 + uint8_t(std::round(((1.0 - sinf(time*0.12f))+1.0f) * 128.0))) % 255;

            bool success = va::read([&capture](cv::UMat& videoFrame){
                //videoFrame will be converted to BGRA and stored in the frameBuffer.
                capture >> videoFrame;
            });

            if(!success)
                break;

            cl::compute([&](cv::UMat& frameBuffer){
                cvtColor(frameBuffer,rgb,cv::COLOR_BGRA2RGB);
                //Color-conversion from RGB to HSV. (OpenCL)
                cv::cvtColor(rgb, hsv, cv::COLOR_RGB2HSV_FULL);
                //Extract the hue channel
                cv::extractChannel(hsv, hueChannel, 0);
                //Set the current hue
                hueChannel.setTo(cvHue);
                //Insert the hue channel
                cv::insertChannel(hueChannel, hsv, 0);
                //Color-conversion from HSV to RGB. (OpenCL)
                cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB_FULL);
                //Color-conversion from RGB to BGRA. (OpenCL)
                cv::cvtColor(rgb, frameBuffer, cv::COLOR_RGB2BGRA);
            });

            //Render using nanovg
            nvg::render([&](NVGcontext* vg, int w, int h) {
                drawColorwheel(vg, w - 300, h - 300, 250.0f, 250.0f, nvgHue);
            });

            va::write([&writer](const cv::UMat& videoFrame){
                //videoFrame is the frameBuffer converted to BGR. Ready to be written.
                writer << videoFrame;
            });

            app::update_fps();

            //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
            if(!app::display())
                break;
        }

        app::terminate();
    });
    return 0;
}
