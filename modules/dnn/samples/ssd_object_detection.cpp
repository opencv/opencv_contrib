#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <iomanip>

using namespace std;

const size_t width = 300;
const size_t height = 300;

Mat getMean(const size_t& imageHeight, const size_t& imageWidth)
{
    Mat mean;

    const int meanValues[3] = {104, 117, 123};
    vector<Mat> meanChannels;
    for(size_t i = 0; i < 3; i++)
    {
        Mat channel(imageHeight, imageWidth, CV_32F, Scalar(meanValues[i]));
        meanChannels.push_back(channel);
    }
    cv::merge(meanChannels, mean);
    return mean;
}

Mat preprocess(const Mat& frame)
{
    Mat preprocessed;
    frame.convertTo(preprocessed, CV_32FC3);
    resize(preprocessed, preprocessed, Size(width, height)); //SSD accepts 300x300 RGB-images

    Mat mean = getMean(width, height);
    cv::subtract(preprocessed, mean, preprocessed);

    return preprocessed;
}

cv::Rect getObjectRectangle(const cv::Mat& detectionMat, const size_t& row,
                            const size_t& frameRows, const size_t& frameCols)
{
    float xLeftBottom = detectionMat.at<float>(row, 3) * frameCols;
    float yLeftBottom = detectionMat.at<float>(row, 4) * frameRows;
    float xRightTop = detectionMat.at<float>(row, 5) * frameCols;
    float yRightTop = detectionMat.at<float>(row, 6) * frameRows;

    return Rect(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);
}

std::string getPascalClassName(const size_t& classId)
{
    switch(classId)
    {
        case 0: return "background";
        case 1: return "aeroplane";
        case 2: return "bicycle";
        case 3: return "bird";
        case 4: return "boat";
        case 5: return "bottle";
        case 6: return "bus";
        case 7: return "car";
        case 8: return "cat";
        case 9: return "chair";
        case 10: return "cow";
        case 11: return "diningtable";
        case 12: return "dog";
        case 13: return "horse";
        case 14: return "motorbike";
        case 15: return "person";
        case 16: return "pottedplant";
        case 17: return "sheep";
        case 18: return "sofa";
        case 19: return "train";
        case 20: return "tvmonitor";
        default: return "wrong label";
    }
}

namespace
{
int bitget(const int& byteval, const int& idx)
{
    return ((byteval & (1 << idx)) != 0);
}
}

Scalar getPascalClassColor(const size_t& classId)
{
    int c = classId;
    int r = 0, g = 0, b = 0;
    for(size_t j = 0; j < 8; j++)
    {
        r = r | (bitget(c, 0) << 7-j);
        g = g | (bitget(c, 1) << 7-j);
        b = b | (bitget(c, 2) << 7-j);
        c = c >> 3;
    }

    return Scalar(r, g, b);
}

const char* about = "This sample uses Single-Shot Detector "
                    "(https://arxiv.org/abs/1512.02325)"
                    "to detect objects on image\n"; // TODO: link

const char* params
    = "{ help           | false | print usage         }"
      "{ proto          |       | model configuration }"
      "{ model          |       | model weights       }"
      "{ image          |       | image for detection }"
      "{ min_confidence | 0.5   | min confidence      }";

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, params);

    if (parser.get<bool>("help") || argc < 4)
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
    Mat preprocessedFrame = preprocess(frame);

    dnn::Blob inputBlob = dnn::Blob(preprocessedFrame); //Convert Mat to dnn::Blob image
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

    float confidenceThreshold = parser.get<float>("min_confidence");
    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            size_t objectClass = detectionMat.at<float>(i, 1);
            Rect object = getObjectRectangle(detectionMat, i, frame.rows, frame.cols);

            rectangle(frame, object, getPascalClassColor(objectClass), 2);

            std::ostringstream text;
            text << getPascalClassName(objectClass) << " " << std::setprecision(3) << confidence;

            Point descriptionPoint(object.x + 5, object.y + 15);
            putText(frame, text.str(), descriptionPoint, cv::FONT_HERSHEY_PLAIN,
                    1, getPascalClassColor(objectClass), 2);
        }
    }

    imshow("detections", frame);
    waitKey();

    return 0;
} // main
