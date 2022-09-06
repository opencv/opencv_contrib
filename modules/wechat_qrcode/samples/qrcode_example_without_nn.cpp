#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

#include <opencv2/wechat_qrcode.hpp>

int main(int argc, char* argv[]) {
    cout << endl << argv[0] << endl << endl;
    cout << "A demo program of WeChat QRCode Detector: " << endl;

    Mat img;
    int camIdx = -1;
    if (argc > 1) {
        bool live = strcmp(argv[1], "-camera") == 0;
        if (live) {
            camIdx = argc > 2 ? atoi(argv[2]) : 0;
        } else {
            img = imread(argv[1]);
        }
    } else {
        cout << "    Usage: " << argv[0] << " <input_image>" << endl;
        return 0;
    }
    // The model is downloaded to ${CMAKE_BINARY_DIR}/downloads/wechat_qrcode if cmake runs without warnings,
    // otherwise you can download them from https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode.
    Ptr<wechat_qrcode::WeChatQRCode> detector;

    try {
        detector = makePtr<wechat_qrcode::WeChatQRCode>("", "", "", "");
    } catch (const std::exception& e) {
        cout <<
            "\n---------------------------------------------------------------\n"
            "Failed to initialize WeChatQRCode.\n"
            "---------------------------------------------------------------\n";
        cout << e.what() << endl;
        return 0;
    }
    string prevstr = "";
    vector<Mat> points;

    if (camIdx < 0) {
        auto res = detector->detectAndDecode(img, points);
        for (const auto& t : res) cout << t << endl;
    } else {
        VideoCapture cap(camIdx);
        for(;;) {
            cap >> img;
            if (img.empty())
                break;
            auto res = detector->detectAndDecode(img, points);
            for (const auto& t : res) {
                if (t != prevstr)
                    cout << t << endl;
            }
            if (!res.empty())
                prevstr = res.back();
            imshow("image", img);
            if (waitKey(30) >= 0)
                break;
        }
    }
    return 0;
}