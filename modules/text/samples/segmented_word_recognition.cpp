/*
 * segmented_word_recognition.cpp
 *
 * A demo program on segmented word recognition.
 * Shows the use of the OCRHMMDecoder API with the two provided default character classifiers.
 *
 * Created on: Jul 31, 2015
 *     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
 */

#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;
using namespace text;


int main(int argc, char* argv[]) {

    const String keys =
      "{help h usage ? |      | print this message.}"
      "{@image         |      | source image for recognition.}"
      "{@mask          |      | binary segmentation mask where each contour is a character.}"
      "{lexicon lex l  |      | (optional) lexicon provided as a list of comma separated words.}"
      ;
    CommandLineParser parser(argc, argv, keys);

    parser.about("\nSegmented word recognition.\nA demo program on segmented word recognition. Shows the use of the OCRHMMDecoder API with the two provided default character classifiers.\n");

    String filename1 = parser.get<String>(0);
    String filename2 = parser.get<String>(1);

    parser.printMessage();
    cout << endl << endl;
    if ((parser.has("help")) || (filename1.size()==0))
    {
        return 0;
    }
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    Mat image = imread(filename1);
    Mat mask;
    if (filename2.size() > 0)
      mask = imread(filename2);
    else
      image.copyTo(mask);

    // be sure the mask is a binary image
    cvtColor(mask, mask, COLOR_BGR2GRAY);
    threshold(mask, mask, 128., 255, THRESH_BINARY);

    // character recognition vocabulary
    string voc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    // Emission probabilities for the HMM language model (identity matrix by default)
    Mat emissionProbabilities = Mat::eye((int)voc.size(), (int)voc.size(), CV_64FC1);
    // Bigram transition probabilities for the HMM language model
    Mat transitionProbabilities;

    string lex = parser.get<string>("lex");
    if (lex.size()>0)
    {
        // Build tailored language model for the provided lexicon
        vector<string> lexicon;
        size_t pos = 0;
        string delimiter = ",";
        std::string token;
        while ((pos = lex.find(delimiter)) != std::string::npos) {
            token = lex.substr(0, pos);
            lexicon.push_back(token);
            lex.erase(0, pos + delimiter.length());
        }
        lexicon.push_back(lex);
        createOCRHMMTransitionsTable(voc,lexicon,transitionProbabilities);
    } else {
        // Or load the generic language model (from Aspell English dictionary)
        FileStorage fs("./OCRHMM_transitions_table.xml", FileStorage::READ);
        fs["transition_probabilities"] >> transitionProbabilities;
        fs.release();
    }

    Ptr<OCRTesseract>  ocrTes = OCRTesseract::create();

    Ptr<OCRHMMDecoder> ocrNM  = OCRHMMDecoder::create(
                                 loadOCRHMMClassifierNM("./OCRHMM_knn_model_data.xml.gz"),
                                 voc, transitionProbabilities, emissionProbabilities);

    Ptr<OCRHMMDecoder> ocrCNN = OCRHMMDecoder::create(
                                 loadOCRHMMClassifierCNN("OCRBeamSearch_CNN_model_data.xml.gz"),
                                 voc, transitionProbabilities, emissionProbabilities);

    std::string output;
    double t_r = (double)getTickCount();
    ocrTes->run(mask, output);
    output.erase(remove(output.begin(), output.end(), '\n'), output.end());
    cout << " OCR_Tesseract  output \"" << output << "\". Done in "
         << ((double)getTickCount() - t_r)*1000/getTickFrequency() << " ms." << endl;

    t_r = (double)getTickCount();
    ocrNM->run(mask, output);
    cout << " OCR_NM         output \"" << output << "\". Done in "
         << ((double)getTickCount() - t_r)*1000/getTickFrequency() << " ms." << endl;

    t_r = (double)getTickCount();
    ocrCNN->run(image, mask, output);
    cout << " OCR_CNN        output \"" << output << "\". Done in "
         << ((double)getTickCount() - t_r)*1000/getTickFrequency() << " ms." << endl;
}
