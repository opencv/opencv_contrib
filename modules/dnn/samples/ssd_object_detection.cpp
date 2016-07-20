#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;


//static void colorizeSegmentation(dnn::Blob& score,
//                                 const vector<cv::Vec3b>& colors,
//                                 cv::Mat& segm)
//{
//    const int rows = score.rows();
//    const int cols = score.cols();
//    const int chns = score.channels();

//    cv::Mat maxCl(rows, cols, CV_8UC1);
//    cv::Mat maxVal(rows, cols, CV_32FC1);
//    for (int ch = 0; ch < chns; ch++)
//    {
//        for (int row = 0; row < rows; row++)
//        {
//            const float* ptrScore = score.ptrf(0, ch, row);
//            uchar* ptrMaxCl = maxCl.ptr<uchar>(row);
//            float* ptrMaxVal = maxVal.ptr<float>(row);
//            for (int col = 0; col < cols; col++)
//            {
//                if (ptrScore[col] > ptrMaxVal[col])
//                {
//                    ptrMaxVal[col] = ptrScore[col];
//                    ptrMaxCl[col] = ch;
//                }
//            }
//        }
//    }

//    segm.create(rows, cols, CV_8UC3);
//    for (int row = 0; row < rows; row++)
//    {
//        const uchar* ptrMaxCl = maxCl.ptr<uchar>(row);
//        cv::Vec3b* ptrSegm = segm.ptr<cv::Vec3b>(row);
//        for (int col = 0; col < cols; col++)
//        {
//            ptrSegm[col] = colors[ptrMaxCl[col]];
//        }
//    }
//}

const char* about = "This sample uses Single-Shot Detector to detect objects "
                    "from camera\n"; // TODO: link

const char* params
    = "{ help    | help                | false | print usage         }"
      "{ proto   | model prototxt file |       | model configuration }"
      "{ model   | caffemodel file     |       | model weights       }";

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, params);

    if (parser.get<bool>("help"))
    {
        std::cout << about << std::endl;
        parser.printMessage();
        return 0;
    }

    String modelConfiguration = parser.get<string>("proto");
    String modelBinary = parser.get<string>("model");

    //! [Create the importer of Caffe model]
    Ptr<dnn::Importer> importer;

    // Import Caffe SSD model
    try
    {
        importer = dnn::createCaffeImporter(modelConfiguration, modelBinary);
    }
    catch (const cv::Exception &err) //Importer can throw errors, we will catch them
    {
        cerr << err.msg << endl;
    }
    //! [Create the importer of Caffe model]

    if (!importer)
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "prototxt:   " << modelConfiguration << endl;
        cerr << "caffemodel: " << modelBinary << endl;
        cerr << "Models can be downloaded here:" << endl;
        cerr << "https://github.com/weiliu89/caffe/tree/ssd#models" << endl;
        exit(-1);
    }

    //! [Initialize network]
    dnn::Net net;
    importer->populateNet(net);
    importer.release();          //We don't need importer anymore
    //! [Initialize network]

    VideoCapture camera;
    if (!camera.open(0))
    {
        cout << "Unable to open camera stream" << endl;
        return 0;
    }

    size_t i = 0;
    for (;; )
    {
        Mat frame;
        camera >> frame;
        if (frame.empty())
            break;

        //! [Prepare blob]
        resize(frame, frame, Size(300, 300));       //SSD accepts 300x300 RGB-images
        dnn::Blob inputBlob = dnn::Blob(frame);       //Convert Mat to dnn::Blob image
        //! [Prepare blob]

        std::ostringstream stream;
        stream << "folder/" << i << ".jpg";
        imwrite(stream.str(), frame);

        //! [Set input blob]
        net.setBlob(".data", inputBlob);            //set the network input
        //! [Set input blob]

        //! [Make forward pass]
        net.forward();                              //compute output
        //! [Make forward pass]

//        //! [Gather output]
//        dnn::Blob detection = net.getBlob("detection_out");

//        //    cv::Mat colorize;
//        //    colorizeSegmentation(score, colors, colorize);
//        //    cv::Mat show;
//        //    cv::addWeighted(img, 0.4, colorize, 0.6, 0.0, show);
//        //    cv::imshow("show", show);
//        //    cv::waitKey(0);
//        //    return 0;

//        imshow("frame", frame);
//        if (waitKey(1) == 27)
//            break;  // stop capturing by pressing ESC
    }

    camera.release();
} // main
