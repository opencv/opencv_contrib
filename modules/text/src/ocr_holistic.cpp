// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

#include <fstream>

using namespace std;

namespace cv { namespace text {

class OCRHolisticWordRecognizerImpl CV_FINAL : public OCRHolisticWordRecognizer
{
private:
    dnn::Net net;
    vector<string> words;

public:
    OCRHolisticWordRecognizerImpl(const string &archFilename, const string &weightsFilename, const string &wordsFilename)
    {
        net = dnn::readNetFromCaffe(archFilename, weightsFilename);
        std::ifstream in(wordsFilename.c_str());
        if (!in)
        {
            CV_Error(Error::StsError, "Could not read Labels from file");
        }
        std::string line;
        while (std::getline(in, line))
            words.push_back(line);
        CV_Assert(getClassCount() == words.size());
    }

    void run(Mat& image, std::string& output_text, std::vector<Rect>* component_rects=NULL, std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL, int component_level=0) CV_OVERRIDE
    {
        CV_Assert(component_level==OCR_LEVEL_WORD); //Componnents not applicable for word spotting
        double confidence;
        output_text = classify(image, confidence);
        if(component_rects!=NULL){
            component_rects->resize(1);
            (*component_rects)[0]=Rect(0,0,image.size().width,image.size().height);
        }
        if(component_texts!=NULL){
            component_texts->resize(1);
            (*component_texts)[0] = output_text;
        }
        if(component_confidences!=NULL){
            component_confidences->resize(1);
            (*component_confidences)[0] = float(confidence);
        }
    }

    void run(Mat& image, Mat& mask, std::string& output_text, std::vector<Rect>* component_rects=NULL, std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL, int component_level=0) CV_OVERRIDE
    {
        //Mask is ignored because the CNN operates on a full image
        CV_Assert(mask.cols == image.cols && mask.rows == image.rows);
        this->run(image, output_text, component_rects, component_texts, component_confidences, component_level);
    }

protected:
    Size getPerceptiveField() const
    {
        return Size(100, 32);
    }

    size_t getClassCount()
    {
        int id = net.getLayerId("prob");
        dnn::MatShape inputShape;
        inputShape.push_back(1);
        inputShape.push_back(1);
        inputShape.push_back(getPerceptiveField().height);
        inputShape.push_back(getPerceptiveField().width);
        vector<dnn::MatShape> inShapes, outShapes;
        net.getLayerShapes(inputShape, id, inShapes, outShapes);
        CV_Assert(outShapes.size() == 1 && outShapes[0].size() == 4);
        CV_Assert(outShapes[0][0] == 1 && outShapes[0][2] == 1 && outShapes[0][3] == 1);
        return outShapes[0][1];
    }

    string classify(InputArray image, double & conf)
    {
        CV_Assert(image.channels() == 1 && image.depth() == CV_8U);
        Mat resized;
        resize(image, resized, getPerceptiveField(), 0, 0, INTER_LINEAR_EXACT);
        Mat blob = dnn::blobFromImage(resized);
        net.setInput(blob, "data");
        Mat prob = net.forward("prob");
        CV_Assert(prob.dims == 4 && !prob.empty() && prob.size[1] == (int)getClassCount());
        int idx[4] = {0};
        minMaxIdx(prob, 0, &conf, 0, idx);
        CV_Assert(0 <= idx[1] && idx[1] < (int)words.size());
        return words[idx[1]];
    }

};

Ptr<OCRHolisticWordRecognizer> OCRHolisticWordRecognizer::create(const string &archFilename, const string &weightsFilename, const string &wordsFilename)
{
    return makePtr<OCRHolisticWordRecognizerImpl>(archFilename, weightsFilename, wordsFilename);
}

}} // cv::text::
