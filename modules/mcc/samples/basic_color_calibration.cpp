#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/mcc.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace mcc;

#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/mcc.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace mcc;

const char *about = "Basic color calibration using simple 3x3 linear color matrix";
const char *keys = {
    "{ help h usage ? |    | show this message }"
    "{t       | 0   | chartType: 0-Standard, 1-DigitalSG, 2-Vinyl, default:0}"
    "{m       |     | File path of model, if you don't have the model you can \
                      find the link in the documentation}"
    "{pb      |     | File path of pbtxt file, available along with with the model \
                      file }"
    "{v       |     | Input from video file, if ommited, input comes from camera }"
    "{ci      | 0   | Camera id if input doesnt come from video (-v) }"
    "{nc      | 1   | Maximum number of charts in the image }"
    "{use_gpu |     | Add this flag if you want to use gpu}"};

int main(int argc, char *argv[])
{
    // ----------------------------------------------------------
    // Scroll down a bit (~50 lines) to find actual relevant code
    // ----------------------------------------------------------

    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if (parser.has("help"))
    {
        parser.printMessage();
        return -1;
    }

    int t = parser.get<int>("t");

    CV_Assert(0 <= t && t <= 2);
    TYPECHART chartType = TYPECHART(t);

    string model_path = parser.get<string>("m");
    string pbtxt_path = parser.get<string>("pb");

    int camId = parser.get<int>("ci");
    int nc = parser.get<int>("nc");

    String video;

    if (parser.has("v"))
        video = parser.get<String>("v");

    bool use_gpu = parser.has("use_gpu");

    if (!parser.check())
    {
        parser.printErrors();
        parser.printMessage();
        return 0;
    }

    VideoCapture inputVideo;
    int waitTime;
    if (!video.empty())
    {
        inputVideo.open(video);
        waitTime = 10;
    }
    else
    {
        inputVideo.open(camId);
        waitTime = 10;
    }

    //--------------------------------------------------------------------------
    //-------------------------Actual Relevant Code-----------------------------
    //--------------------------------------------------------------------------

    //load the network

    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model_path, pbtxt_path);

    if (use_gpu)
    {
        net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
    }

    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    if (!detector->setNet(net))
    {
        cout << "Loading Model failed: Aborting" << endl;
        return 0;
    }
    namedWindow("original", WINDOW_NORMAL);
    namedWindow("image result | q or esc to quit", WINDOW_NORMAL);

    while (inputVideo.grab())
    {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        imageCopy = image.clone();

        // Marker type to detect
        if (!detector->process(image, chartType, nc, true))
        {
            printf("ChartColor not detected \n");
        }
        else
        {

            // get checker
            std::vector<Ptr<mcc::CChecker>> checkers = detector->getListColorChecker();
            for (Ptr<mcc::CChecker> checker : checkers)
            {
                // current checker
                Ptr<CCheckerDraw> cdraw = CCheckerDraw::create(checker);
                cdraw->draw(image);
                Mat a = checker->getChartsRGB();
                Mat b = checker->getActualChartsColors();

                a = a(Range::all(), Range(1, 2)).clone(); // second column contains the mean of each color
                a = a.reshape(1, a.rows/3);
                a.convertTo(a, CV_32F );

                b = b(Range::all(), Range(0,3)).clone(); // The first 3 are rgb

                Mat x;
                cv::solve(a, b, x,DECOMP_SVD);

                int originalRows= image.rows;

                cvtColor(image, image, COLOR_BGR2RGB);
                image.convertTo(image, CV_32FC3);
                image = image.reshape(1, image.rows*image.cols);

                image = image * x;

                image = image.reshape(3, originalRows);
                image.convertTo(image, CV_8UC3);
                cvtColor(image, image, COLOR_RGB2BGR);

            }
        }

        imshow("image result | q or esc to quit", image);
        imshow("original", imageCopy);
        char key = (char)waitKey(waitTime);
        if (key == 27)
            break;
    }

    return 0;
}
