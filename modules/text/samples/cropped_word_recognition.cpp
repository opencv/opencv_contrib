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

    string vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; // must have the same order as the clasifier output classes
    vector<string> lexicon;  // a list of words expected to be found on the input image
    lexicon.push_back(string("abb"));
    lexicon.push_back(string("riser"));
    lexicon.push_back(string("CHINA"));
    lexicon.push_back(string("HERE"));
    lexicon.push_back(string("President"));
    lexicon.push_back(string("smash"));
    lexicon.push_back(string("KUALA"));
    lexicon.push_back(string("Produkt"));
    lexicon.push_back(string("NINTENDO"));

    // Create tailored language model a small given lexicon
    Mat transition_p;
    createOCRHMMTransitionsTable(vocabulary,lexicon,transition_p);

    // An alternative would be to load the default generic language model
    //    (created from ispell 42869 english words list)
    /*Mat transition_p;
    string filename = "OCRHMM_transitions_table.xml";
    FileStorage fs(filename, FileStorage::READ);
    fs["transition_probabilities"] >> transition_p;
    fs.release();*/

    Mat emission_p = Mat::eye(62,62,CV_64FC1);

    // Notice we set here a beam size of 50. This is much faster than using the default value (500).
    // 50 works well with our tiny lexicon example, but may not with larger dictionaries.
    Ptr<OCRBeamSearchDecoder> ocr = OCRBeamSearchDecoder::create(
                loadOCRBeamSearchClassifierCNN("OCRBeamSearch_CNN_model_data.xml.gz"),
                vocabulary, transition_p, emission_p, OCR_DECODER_VITERBI, 50);

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
