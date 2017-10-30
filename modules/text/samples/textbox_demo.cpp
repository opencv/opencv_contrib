#include <opencv2/text.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include  <sstream>
#include  <iostream>
#include  <fstream>

using namespace cv;

namespace
{
std::string getHelpStr(const std::string& progFname)
{
    std::stringstream out;
    out << "    Demo of text detection CNN for text detection." << std::endl
        << "    Minghui Liao, Baoguang Shi, Xiang Bai, Xinggang Wang, Wenyu Liu: TextBoxes: A Fast Text Detector with a Single Deep Neural Network, AAAI2017\n\n"
        << "    Usage: " << progFname << " <output_file> <input_image>" << std::endl
        << "    Caffe Model files  (textbox.prototxt, TextBoxes_icdar13.caffemodel)"<<std::endl
        << "      must be in the current directory. See the documentation of text::TextDetectorCNN class to get download links." << std::endl;
    return out.str();
}

bool fileExists (const std::string& filename)
{
    std::ifstream f(filename.c_str());
    return f.good();
}

void textbox_draw(Mat src, std::vector<Rect>& groups, std::vector<float>& probs, std::vector<int>& indexes)
{
    for (size_t i = 0; i < indexes.size(); i++)
    {
        if (src.type() == CV_8UC3)
        {
            Rect currrentBox = groups[indexes[i]];
            rectangle(src, currrentBox, Scalar( 0, 255, 255 ), 2, LINE_AA);
            String label = format("%.2f", probs[indexes[i]]);
            std::cout << "text box: " << currrentBox << " confidence: " << probs[indexes[i]] << "\n";

            int baseLine = 0;
            Size labelSize = getTextSize(label, FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
            int yLeftBottom = std::max(currrentBox.y, labelSize.height);
            rectangle(src, Point(currrentBox.x, yLeftBottom - labelSize.height),
                      Point(currrentBox.x + labelSize.width, yLeftBottom + baseLine), Scalar( 255, 255, 255 ), FILLED);

            putText(src, label, Point(currrentBox.x, yLeftBottom), FONT_HERSHEY_PLAIN, 1, Scalar( 0,0,0 ), 1, LINE_AA);
        }
        else
            rectangle(src, groups[i], Scalar( 255 ), 3, 8 );
    }
}

}

int main(int argc, const char * argv[])
{
    if (argc < 2)
    {
        std::cout << getHelpStr(argv[0]);
        std::cout << "Insufiecient parameters. Aborting!" << std::endl;
        exit(1);
    }

    const std::string modelArch = "textbox.prototxt";
    const std::string moddelWeights = "TextBoxes_icdar13.caffemodel";

    if (!fileExists(modelArch) || !fileExists(moddelWeights))
    {
        std::cout << getHelpStr(argv[0]);
        std::cout << "Model files not found in the current directory. Aborting!" << std::endl;
        exit(1);
    }

    Mat image = imread(String(argv[1]), IMREAD_COLOR);

    std::cout << "Starting Text Box Demo" << std::endl;
    Ptr<text::TextDetectorCNN> textSpotter =
            text::TextDetectorCNN::create(modelArch, moddelWeights);

    std::vector<Rect> bbox;
    std::vector<float> outProbabillities;
    textSpotter->detect(image, bbox, outProbabillities);

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(bbox, outProbabillities, 0.3f, 0.4f, indexes);

    textbox_draw(image, bbox, outProbabillities, indexes);

    imshow("TextBox Demo",image);
    std::cout << "Done!" << std::endl << std::endl;
    std::cout << "Press any key to exit." << std::endl << std::endl;
    waitKey();
    return 0;
}
