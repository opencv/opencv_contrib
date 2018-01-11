#include  "opencv2/text.hpp"
#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"

#include  <sstream>
#include  <iostream>

using namespace std;
using namespace cv;
using namespace cv::text;

inline void printHelp()
{
    cout << "    Demo of wordspotting CNN for text recognition." << endl;
    cout << "    Max Jaderberg et al.: Reading Text in the Wild with Convolutional Neural Networks, IJCV 2015"<<std::endl<<std::endl;

    cout << "    Usage: program <input_image>" << endl;
    cout << "    Caffe Model files  (dictnet_vgg.caffemodel, dictnet_vgg_deploy.prototxt, dictnet_vgg_labels.txt)"<<endl;
    cout << "      must be in the current directory." << endl << endl;

    cout << "    Obtaining Caffe Model files in linux shell:"<<endl;
    cout << "    wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg.caffemodel"<<endl;
    cout << "    wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_deploy.prototxt"<<endl;
    cout << "    wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_labels.txt"<<endl<<endl;
}

int main(int argc, const char * argv[])
{
    if (argc != 2)
    {
        printHelp();
        exit(1);
    }

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);

    cout << "Read image (" << argv[1] << "): " << image.size << ", channels: " << image.channels() << ", depth: " << image.depth() << endl;

    if (image.empty())
    {
        printHelp();
        exit(1);
    }

    Ptr<OCRHolisticWordRecognizer> wordSpotter = OCRHolisticWordRecognizer::create("dictnet_vgg_deploy.prototxt", "dictnet_vgg.caffemodel", "dictnet_vgg_labels.txt");

    std::string word;
    vector<float> confs;
    wordSpotter->run(image, word, 0, 0, &confs);

    cout << "Detected word: '" << word << "', confidence: " << confs[0] << endl;
}
