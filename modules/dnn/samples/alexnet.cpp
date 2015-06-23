#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

int main(void)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter("bvlc_alexnet.prototxt", "bvlc_alexnet.caffemodel");
        importer->populateNet(net);
    }

    Mat img = imread("alexnet.png");
    CV_Assert(!img.empty());
    img.convertTo(img, CV_32F, 1.0 / 255);
    Blob imgBlob(img);

    net.setBlob("data", imgBlob);

    net.forward();

    Blob res = net.getBlob("prob");
    
    return 0;
}