#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

int main() {
    Ptr<V4D> window = V4D::make(Size(1280, 720), cv::Size(), "Vector Graphics");

    //Display the framebuffer in the native window in an endless loop
    window->run([=]() {
        //Creates a NanoVG context and draws eyes
        window->nvg([](const Size& sz) {
            //Calls from this namespace may only be used inside a nvg context
            using namespace cv::v4d::nvg;
            clear();
            float t = cv::getTickCount();
            float x = 0;
            float y = 0;
            float w = sz.width / 4;
            float h = sz.height / 4;
            float mx = w / 2.0;
            float my = h / 2.0;
            Paint gloss, bg;
            float ex = w * 0.23f;
            float ey = h * 0.5f;
            float lx = x + ex;
            float ly = y + ey;
            float rx = x + w - ex;
            float ry = y + ey;
            float dx, dy, d;
            float br = (ex < ey ? ex : ey) * 0.5f;
            float blink = 1 - pow(sinf(t * 0.5f), 200) * 0.8f;

            bg = linearGradient(x, y + h * 0.5f, x + w * 0.1f, y + h, cv::Scalar(0, 0, 0, 32), cv::Scalar(0,0,0,16));
            beginPath();
            ellipse(lx + 3.0f, ly + 16.0f, ex, ey);
            ellipse(rx + 3.0f, ry + 16.0f, ex, ey);
            fillPaint(bg);
            fill();

            bg = linearGradient(x, y + h * 0.25f, x + w * 0.1f, y + h,
                    cv::Scalar(220, 220, 220, 255), cv::Scalar(128, 128, 128, 255));
            beginPath();
            ellipse(lx, ly, ex, ey);
            ellipse(rx, ry, ex, ey);
            fillPaint(bg);
            fill();

            dx = (mx - rx) / (ex * 10);
            dy = (my - ry) / (ey * 10);
            d = sqrtf(dx * dx + dy * dy);
            if (d > 1.0f) {
                dx /= d;
                dy /= d;
            }
            dx *= ex * 0.4f;
            dy *= ey * 0.5f;
            beginPath();
            ellipse(lx + dx, ly + dy + ey * 0.25f * (1 - blink), br, br * blink);
            fillColor(cv::Scalar(32, 32, 32, 255));
            fill();

            dx = (mx - rx) / (ex * 10);
            dy = (my - ry) / (ey * 10);
            d = sqrtf(dx * dx + dy * dy);
            if (d > 1.0f) {
                dx /= d;
                dy /= d;
            }
            dx *= ex * 0.4f;
            dy *= ey * 0.5f;
            beginPath();
            ellipse(rx + dx, ry + dy + ey * 0.25f * (1 - blink), br, br * blink);
            fillColor(cv::Scalar(32, 32, 32, 255));
            fill();

            gloss = radialGradient(lx - ex * 0.25f, ly - ey * 0.5f, ex * 0.1f, ex * 0.75f,
                    cv::Scalar(255, 255, 255, 128), cv::Scalar(255, 255, 255, 0));
            beginPath();
            ellipse(lx, ly, ex, ey);
            fillPaint(gloss);
            fill();

            gloss = radialGradient(rx - ex * 0.25f, ry - ey * 0.5f, ex * 0.1f, ex * 0.75f,
                    cv::Scalar(255, 255, 255, 128), cv::Scalar(255, 255, 255, 0));
            beginPath();
            ellipse(rx, ry, ex, ey);
            fillPaint(gloss);
            fill();
        });

        return window->display();
    });
}

