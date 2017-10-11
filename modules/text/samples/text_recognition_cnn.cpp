#include <opencv2/text.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include  <iostream>
#include  <fstream>

using namespace cv;
using namespace std;

namespace
{
void printHelpStr(const string& progFname)
{
    cout << "   Demo of text recognition CNN for text detection." << endl
         << "   Max Jaderberg et al.: Reading Text in the Wild with Convolutional Neural Networks, IJCV 2015"<<endl<<endl
         << "   Usage: " << progFname << " <output_file> <input_image>" << endl
         << "   Caffe Model files (textbox.prototxt, TextBoxes_icdar13.caffemodel)"<<endl
         << "     must be in the current directory. See the documentation of text::TextDetectorCNN class to get download links." << endl
         << "   Obtaining text recognition Caffe Model files in linux shell:" << endl
         << "   wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg.caffemodel" << endl
         << "   wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_deploy.prototxt" << endl
         << "   wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_labels.txt" <<endl << endl;
}

bool fileExists (const string& filename)
{
    ifstream f(filename.c_str());
    return f.good();
}

void textbox_draw(Mat src, vector<Rect>& groups, vector<float>& probs, float thres)
{
    for (size_t i = 0; i < groups.size(); i++)
    {
        if(probs[i] > thres)
        {
            if (src.type() == CV_8UC3)
            {
                rectangle(src, groups[i], Scalar( 0, 255, 255 ), 2, LINE_AA);
                String label = format("%.2f", probs[i]);
                cout << "text box: " << groups[i] << " confidence: " << probs[i] << "\n";
                putText(src, label, groups.at(i).tl(), FONT_HERSHEY_PLAIN, 1, Scalar( 0,0,255 ), 1, LINE_AA);
            }
            else
                rectangle(src, groups[i], Scalar( 255 ), 3, 8 );
        }
    }
}

}

int main(int argc, const char * argv[])
{
    if (argc < 2)
    {
        printHelpStr(argv[0]);
        cout << "Insufiecient parameters. Aborting!" << endl;
        exit(1);
    }

    const string modelArch = "textbox.prototxt";
    const string moddelWeights = "TextBoxes_icdar13.caffemodel";

    if (!fileExists(modelArch) || !fileExists(moddelWeights))
    {
        printHelpStr(argv[0]);
        cout << "Model files not found in the current directory. Aborting!" << endl;
        exit(1);
    }

    Mat image = imread(String(argv[1]), IMREAD_COLOR);

    cout << "Starting Text Box Demo" << endl;
    Ptr<text::TextDetectorCNN> textSpotter =
            text::TextDetectorCNN::create(modelArch, moddelWeights, false);

    vector<Rect> bbox;
    vector<float> outProbabillities;
    textSpotter->detect(image, bbox, outProbabillities);

    float prob_threshold = 0.6f;
    Mat image_copy = image.clone();
    textbox_draw(image_copy, bbox, outProbabillities, prob_threshold);
    imshow("Text detection", image_copy);
    image_copy = image.clone();

    Ptr<text::OCRHolisticWordRecognizer> wordSpotter =
            text::OCRHolisticWordRecognizer::create("dictnet_vgg_deploy.prototxt", "dictnet_vgg.caffemodel", "dictnet_vgg_labels.txt");

    for(size_t i = 0; i < bbox.size(); i++)
    {
        if(outProbabillities[i] > prob_threshold)
        {
            Mat wordImg;
            cvtColor(image(bbox[i]), wordImg, COLOR_BGR2GRAY);
            string word;
            vector<float> confs;
            wordSpotter->run(wordImg, word, NULL, NULL, &confs);
            rectangle(image_copy, bbox[i], Scalar(0, 255, 255), 1, LINE_AA);
            putText(image_copy, word, bbox[i].tl(), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255), 1, LINE_AA);
        }
    }
    imshow("Text recognition", image_copy);
    cout << "Recognition finished. Press any key to exit.\n";
    waitKey();
    return 0;
}

