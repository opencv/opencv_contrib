#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <fstream>
using namespace cv;
using namespace cv::dnn;

typedef std::pair<int, double> ClassProb;

ClassProb getMaxClass(Blob &probBlob, int sampleNum = 0)
{
    int numClasses = (int)probBlob.total(1);
    Mat probMat(1, numClasses, CV_32F, probBlob.ptr<float>(sampleNum));

    double prob;
    Point probLoc;
    minMaxLoc(probMat, NULL, &prob, NULL, &probLoc);

    return std::make_pair(probLoc.x, prob);
}

std::vector<String> CLASES_NAMES;

void initClassesNames()
{
    std::ifstream fp("ILSVRC2012_synsets.txt");
    CV_Assert(fp.is_open());

    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        CLASES_NAMES.push_back(name);
    }

    CV_Assert(CLASES_NAMES.size() == 1000);

    fp.close();
}


int main(void)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter("bvlc_alexnet.prototxt", "bvlc_alexnet.caffemodel");
        importer->populateNet(net);
    }

    Mat img = imread("zebra.jpg");
    CV_Assert(!img.empty());
    cvtColor(img, img, COLOR_BGR2RGB);
    img.convertTo(img, CV_32F);
    resize(img, img, Size(227, 227));
    subtract(img, cv::mean(img), img);
    Blob imgBlob(img);

    net.setBlob("data", imgBlob);
    net.forward();

    Blob probBlob = net.getBlob("prob");
    ClassProb bc = getMaxClass(probBlob);

    initClassesNames();
    std::string className = (bc.first < (int)CLASES_NAMES.size()) ? CLASES_NAMES[bc.first] : "unnamed";

    std::cout << "Best class:";
    std::cout << " #" << bc.first;
    std::cout << " (from " << probBlob.total(1) << ")";
    std::cout << " \"" + className << "\"";
    std::cout <<  std::endl;
    std::cout << "Prob: " << bc.second * 100 << "%" << std::endl;

    return 0;
}