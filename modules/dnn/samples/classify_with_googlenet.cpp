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
    std::ifstream fp("synset_words.txt");
    CV_Assert(fp.is_open());

    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            CLASES_NAMES.push_back( name.substr(name.find(' ')+1) );
    }

    fp.close();
}


int main(int argc, char **argv)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter("bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel");
        importer->populateNet(net);
    }

    String filename = (argc > 1) ? argv[1] : "space_shuttle.jpg";

    Mat img = imread(filename);
    CV_Assert(!img.empty());
    cvtColor(img, img, COLOR_BGR2RGB);
    resize(img, img, Size(227, 227));
    Blob imgBlob(img);

    net.setBlob(".data", imgBlob);
    net.forward();

    Blob prob = net.getBlob("prob");
    ClassProb bc = getMaxClass(prob);

    initClassesNames();
    std::string className = (bc.first < (int)CLASES_NAMES.size()) ? CLASES_NAMES[bc.first] : "unnamed";

    std::cout << "Best class:";
    std::cout << " #" << bc.first;
    std::cout << " (from " << prob.total(1) << ")";
    std::cout << " \"" + className << "\"";
    std::cout <<  std::endl;
    std::cout << "Prob: " << bc.second * 100 << "%" << std::endl;

    return 0;
}
