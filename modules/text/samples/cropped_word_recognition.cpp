/*
 * textdetection.cpp
 *
 * A demo program of End-to-end Scene Text Detection and Recognition:
 * Shows the use of the Tesseract OCR API with the Extremal Region Filter algorithm described in:
 * Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
 *
 * Created on: Jul 31, 2014
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

//Perform text recognition in a given cropped word
int main(int argc, char* argv[])
{
    cout << endl << argv[0] << endl << endl;
    cout << "A demo program of Scene Text cropped word Recognition: " << endl;
    cout << "Shows the use of the OCRBeamSearchDecoder class using the Single Layer CNN character classifier described in:" << endl;
    cout << "Coates, Adam, et al. \"Text detection and character recognition in scene images with unsupervised feature learning.\" ICDAR 2011." << endl << endl;

    Mat image;
    if(argc>1)
        image  = imread(argv[1]);
    else
    {
        cout << "    Usage: " << argv[0] << " <input_image>" << endl << endl;
        return(0);
    }

    Mat transition_p;
    string filename = "OCRHMM_transitions_table.xml"; // TODO this table was done with a different vocabulary order?
                                                      // TODO add a new function in ocr.cpp to create transition tab
                                                      // for a given lexicon
    FileStorage fs(filename, FileStorage::READ);
    fs["transition_probabilities"] >> transition_p;
    fs.release();
    Mat emission_p = Mat::eye(62,62,CV_64FC1);
    string voc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyx0123456789";

    Ptr<OCRBeamSearchDecoder> ocr = OCRBeamSearchDecoder::create(
                loadOCRBeamSearchClassifierCNN("OCRBeamSearch_CNN_model_data.xml.gz"),
                voc, transition_p, emission_p);

    double t_r = (double)getTickCount();
    string output;

    vector<Rect>   boxes;
    vector<string> words;
    vector<float>  confidences;
    ocr->run(image, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

    cout << "OCR output = \"" << output << "\". Decoded in " 
         << ((double)getTickCount() - t_r)*1000/getTickFrequency() << " ms." << endl << endl;

    return 0;
}
