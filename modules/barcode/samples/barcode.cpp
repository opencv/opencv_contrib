#include <iostream>
#include "opencv2/barcode.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

static int liveBarCodeDetect();

static int imageBarCodeDetect(const string &in_file);

static bool g_detectOnly = false;

static string g_out_file_name, g_out_file_ext;

static Ptr<barcode::BarcodeDetector> bardet;

int main(int argc, char **argv)
{


    const string keys = "{h help ? |        | print help messages }"
                        "{i in     |        | input image path (also switches to image detection mode) }"
                        "{detect   | false  | detect 1D barcode only (skip decoding) }"
                        "{o out    |        | path to result file (only for single image decode) }"
                        "{sr_prototxt|      | super resolution prototxt path }"
                        "{sr_model |        | super resolution model path }";

    CommandLineParser cmd_parser(argc, argv, keys);

    cmd_parser.about("This program detects the 1D barcodes from camera or images using the OpenCV library.");
    if (cmd_parser.has("help"))
    {
        cmd_parser.printMessage();
        return 0;
    }

    string in_file_name = cmd_parser.get<string>("in");    // path to input image
    string sr_prototxt = cmd_parser.get<string>("sr_prototxt");    // path to sr_prototxt
    string sr_model = cmd_parser.get<string>("sr_model");    // path to sr_model
    if (cmd_parser.has("out"))
    {
        std::string fpath = cmd_parser.get<string>("out");   // path to output image
        std::string::size_type idx = fpath.rfind('.');
        if (idx != std::string::npos)
        {
            g_out_file_name = fpath.substr(0, idx);
            g_out_file_ext = fpath.substr(idx);
        }
        else
        {
            g_out_file_name = fpath;
            g_out_file_ext = ".png";
        }
    }
    if (!cmd_parser.check())
    {
        cmd_parser.printErrors();
        return -1;
    }
    g_detectOnly = cmd_parser.has("detect") && cmd_parser.get<bool>("detect");
    //! [initialize]
    try{
        bardet = makePtr<barcode::BarcodeDetector>(sr_prototxt, sr_model);
    } catch (const std::exception& e)
    {
        cout <<
             "\n---------------------------------------------------------------\n"
             "Failed to initialize super resolution.\n"
             "Please, download 'sr.*' from\n"
             "https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode\n"
             "and put them into the current directory.\n"
             "Or you can leave sr_prototxt and sr_model unspecified.\n"
             "---------------------------------------------------------------\n";
        cout << e.what() << endl;
        return -1;
    }
    //! [initialize]
    int return_code;
    if (in_file_name.empty())
    {
        return_code = liveBarCodeDetect();
    }
    else
    {
        return_code = imageBarCodeDetect(in_file_name);
    }
    return return_code;

}

static void drawBarcodeContour(Mat &color_image, const vector<Point> &corners, bool decodable)
{
    if (!corners.empty())
    {
        double show_radius = (color_image.rows > color_image.cols) ? (2.813 * color_image.rows) / color_image.cols :
                             (2.813 * color_image.cols) / color_image.rows;
        double contour_radius = show_radius * 0.4;

        vector<vector<Point> > contours;
        contours.push_back(corners);

        drawContours(color_image, contours, 0, decodable ? Scalar(0, 255, 0) : Scalar(0, 0, 255),
                     cvRound(contour_radius));

        RNG rng(1000);
        for (size_t i = 0; i < 4; i++)
        {
            Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            circle(color_image, corners[i], cvRound(show_radius), color, -1);
        }
    }
}

//! [visualize]
static void drawFPS(Mat &color_image, double fps)
{
    ostringstream convert;
    convert << cv::format("%.2f", fps) << " FPS (" << (g_detectOnly ? " detector" : " decoder") << ")";
    putText(color_image, convert.str(), Point(25, 25), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 2);
}

static void drawBarcodeResults(Mat &frame, const vector<Point> &corners, const vector<cv::String> &decode_info,
                               const vector<cv::barcode::BarcodeType> &decode_type, double fps)
{
    if (!corners.empty())
    {
        for (size_t i = 0; i < corners.size(); i += 4)
        {
            size_t bar_idx = i / 4;
            vector<Point> barcode_contour(corners.begin() + i, corners.begin() + i + 4);
            drawBarcodeContour(frame, barcode_contour, g_detectOnly || decode_type[bar_idx] != barcode::NONE);

            cout << "BAR[" << bar_idx << "] @ " << Mat(barcode_contour).reshape(2, 1) << ": ";
            if (decode_info.size() > bar_idx)
            {
                if (!decode_info[bar_idx].empty())
                {
                    cout << "TYPE: " << decode_type[bar_idx] << " INFO: " << decode_info[bar_idx] << endl;
                }
                else
                {
                    cout << "can't decode 1D barcode" << endl;
                }
            }
            else
            {
                cout << "decode information is not available (disabled)" << endl;
            }
        }
    }
    else
    {
        cout << "Barcode is not detected" << endl;
    }

    drawFPS(frame, fps);
}
//! [visualize]

static void
runBarcode(const Mat &input, vector<Point> &corners, vector<cv::String> &decode_info,
           vector<cv::barcode::BarcodeType> &decode_type
)
{
    if (!g_detectOnly)
    {
        //! [detectAndDecode]
        bool result_detection = bardet->detectAndDecode(input, decode_info, decode_type, corners);
        //! [detectAndDecode]
        CV_UNUSED(result_detection);
    }
    else
    {
        //! [detect]
        bool result_detection = bardet->detect(input, corners);
        //! [detect]
        CV_UNUSED(result_detection);
    }
}

int liveBarCodeDetect()
{
    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "Cannot open a camera" << endl;
        return 2;
    }

    cout << "Press 'd' to switch between decoder and detector" << endl;
    cout << "Press 'ESC' to exit" << endl;

    for (;;)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            cout << "End of video stream" << endl;
            break;
        }


        Mat result;
        if (frame.channels() == 1)
        {
            cvtColor(frame, result, COLOR_GRAY2BGR);
        }
        else
        {
            frame.copyTo(result);
        }
        TickMeter timer;
        //! [output]
        vector<cv::String> decode_info;
        vector<barcode::BarcodeType> decoded_type;
        vector<Point> corners;
        //! [output]
        timer.start();
        runBarcode(frame, corners, decode_info, decoded_type);
        timer.stop();

        double fps = 1 / timer.getTimeSec();
        drawBarcodeResults(result, corners, decode_info, decoded_type, fps);

        if (!result.empty())
        {
            imshow("barcode", result);
        }

        int code = waitKey(1);
        if (code < 0)
        {
            continue;
        } // timeout
        char c = (char) code;

        if (c == 'd')
        {
            g_detectOnly = !g_detectOnly;
            cout << "Switching barcode decoder mode ==> " << (g_detectOnly ? "detect" : "decode") << endl;
        }
        if (c == 27)
        {
            cout << "'ESC' is pressed. Exiting..." << endl;
            break;
        }
    }
    cout << "Exit." << endl;

    return 0;
}

int imageBarCodeDetect(const string &in_file)
{
    const int count_experiments = 10;

    Mat input = imread(in_file, IMREAD_COLOR);
    cout << "Run BarCode" << (g_detectOnly ? " detector" : " decoder") << " on image: " << input.size() << " ("
         << typeToString(input.type()) << ")" << endl;

    vector<Point> corners;
    vector<cv::String> decode_info;
    vector<barcode::BarcodeType> decoded_type;
    TickMeter timer;
    for (size_t i = 0; i < count_experiments; i++)
    {
        corners.clear();
        decode_info.clear();

        timer.start();
        runBarcode(input, corners, decode_info, decoded_type);
        timer.stop();
    }
    double fps = count_experiments / timer.getTimeSec();
    cout << "FPS: " << fps << endl;

    Mat result;
    input.copyTo(result);
    drawBarcodeResults(result, corners, decode_info, decoded_type, fps);
    if (!g_out_file_name.empty())
    {
        string out_file = g_out_file_name + g_out_file_ext;
        cout << "Saving result: " << out_file << endl;
        imwrite(out_file, result);
    }
    imshow("barcode", result);
    waitKey(1);

    cout << "Press any key to exit ..." << endl;
    waitKey(0);
    cout << "Exit." << endl;

    return 0;
}
