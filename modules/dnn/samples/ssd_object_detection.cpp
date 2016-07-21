#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

const size_t width = 300;
const size_t height = 300;

Mat getMean(const size_t& height, const size_t& width)
{
    Mat mean;

    const int meanValues[3] = {104, 117, 123};
    vector<Mat> meanChannels;
    for(size_t i = 0; i < 3; i++)
    {
        Mat channel(height, width, CV_32F, Scalar(meanValues[i]));
        meanChannels.push_back(channel);
    }
    cv::merge(meanChannels, mean);
    return mean;
}

void preprocess(Mat& frame)
{
    frame.convertTo(frame, CV_32FC3);
    resize(frame, frame, Size(width, height)); //SSD accepts 300x300 RGB-images

    Mat mean = getMean(width, height);
    cv::subtract(frame, mean, frame);
}

const char* about = "This sample uses Single-Shot Detector "
                    "(https://arxiv.org/abs/1512.02325)"
                    "to detect objects on image\n"; // TODO: link

const char* params
    = "{ help   | help                | false | print usage         }"
      "{ proto  | model prototxt file |       | model configuration }"
      "{ model  | caffemodel file     |       | model weights       }"
      "{ image  | image file          |       | image for detection }";

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

    cv::Mat frame = cv::imread(parser.get<string>("image"), -1);

    //! [Prepare blob]
    preprocess(frame);

    dnn::Blob inputBlob = dnn::Blob(frame);           //Convert Mat to dnn::Blob image
    //! [Prepare blob]

    //! [Set input blob]
    net.setBlob(".data", inputBlob);                //set the network input
    //! [Set input blob]

    //! [Make forward pass]
    net.forward();                                  //compute output
    //! [Make forward pass]

    //! [Gather output]
    dnn::Blob detection = net.getBlob("detection_out");
    Mat detectionMat(detection.rows(), detection.cols(), CV_32F, detection.ptrf());

    for(size_t i = 0; i < detectionMat.rows; i++)
    {
        std::cout << "Class: " << detectionMat.at<float>(i, 1) << std::endl;
        std::cout << "Confidence: " << detectionMat.at<float>(i, 2) << std::endl;

        std::cout << " " << detectionMat.at<float>(i, 3) * width;
        std::cout << " " << detectionMat.at<float>(i, 4) * height;
        std::cout << " " << detectionMat.at<float>(i, 5) * width;
        std::cout << " " << detectionMat.at<float>(i, 6) * height;

        float xLeftBottom = detectionMat.at<float>(i, 3) * width;
        float yLeftBottom = detectionMat.at<float>(i, 4) * height;
        float xRightTop = detectionMat.at<float>(i, 5) * width;
        float yRightTop = detectionMat.at<float>(i, 6) * height;

        Rect object(xLeftBottom, yLeftBottom,
                    xRightTop - xLeftBottom,
                    yRightTop - yLeftBottom);

        rectangle(frame, object, Scalar(0, 255, 0));
    }

    imshow("detections", frame);
    waitKey();

    return 0;
} // main
