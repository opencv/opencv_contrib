/*
 * cropped_word_recognition.cpp
 *
 * A demo program of text recognition in a given cropped word.
 * Shows the use of the OCRBeamSearchDecoder class API using the provided default classifier.
 *
 * Created on: Jul 9, 2015
 *     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
 */

#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::text;

int main(int argc, char* argv[])
{

    cout << endl << argv[0] << endl << endl;
    cout << "A demo program of Scene Text Character Recognition: " << endl;
    cout << "Shows the use of the OCRBeamSearchDecoder::ClassifierCallback class using the Single Layer CNN character classifier described in:" << endl;
    cout << "Coates, Adam, et al. \"Text detection and character recognition in scene images with unsupervised feature learning.\" ICDAR 2011." << endl << endl;

    Mat image;
    if(argc>1)
        image  = imread(argv[1]);
    else
    {
        cout << "    Usage: " << argv[0] << " <input_image>" << endl;
        cout << "           the input image must contain a single character (e.g. scenetext_char01.jpg)." << endl << endl;
        return(0);
    }

    string vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; // must have the same order as the classifier output classes

    Ptr<OCRHMMDecoder::ClassifierCallback> ocr = loadOCRHMMClassifierCNN("OCRBeamSearch_CNN_model_data.xml.gz");

    double t_r = (double)getTickCount();
    vector<int> out_classes;
    vector<double> out_confidences;

    ocr->eval(image, out_classes, out_confidences);

    cout << "OCR output = \"" << vocabulary[out_classes[0]] << "\" with confidence "
         << out_confidences[0] << ". Evaluated in "
         << ((double)getTickCount() - t_r)*1000/getTickFrequency() << " ms." << endl << endl;

    return 0;
}
