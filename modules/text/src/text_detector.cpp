#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"



#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>


#ifdef HAVE_CAFFE
#include "caffe/caffe.hpp"
#endif

namespace cv { namespace text {




class textDetectImpl: public textDetector{
private:
    struct NetOutput{
        //Auxiliary structure that handles the logic of getting bounding box and confidences of textness from
        //the raw outputs of caffe
        Rect bbox;
        float probability;


        static void getOutputs(const float* buffer,int nbrTextBoxes,int nCol,std::vector<NetOutput>& res,Size inputShape)
        {

            res.resize(nbrTextBoxes);
            for(int k=0;k<nbrTextBoxes;k++)
            {
                float x_min = buffer[k*nCol+3]*inputShape.width;
                float y_min = buffer[k*nCol+4]*inputShape.height;
                float x_max = buffer[k*nCol+5]*inputShape.width;
                float y_max = buffer[k*nCol +6]*inputShape.height;
                x_min = x_min<0?0:x_min;
                y_min = y_min<0?0:y_min;
                x_max = x_max> inputShape.width?inputShape.width-1:x_max;
                y_max = y_max > inputShape.height?inputShape.height-1:y_max;
                float wd = x_max-x_min+1;
                float ht = y_max-y_min+1;

                res[k].bbox=Rect(int(x_min),int(y_min),int(wd),int(ht));

                res[k].probability=buffer[k*nCol+2];
            }

        }


    };
protected:

    Ptr<TextRegionDetector> classifier_;
public:
    textDetectImpl(Ptr<TextRegionDetector> classifierPtr):classifier_(classifierPtr)
    {

    }



    void textDetectInImage(InputArray inputImage,CV_OUT std::vector<Rect>& Bbox,CV_OUT std::vector<float>& confidence)
    {
                Mat netOutput;
                // call the detect function of deepCNN class
                this->classifier_->detect(inputImage,netOutput);
               // get the output geometry i.e height and width of output blob from caffe
                Size OutputGeometry_ = this->classifier_->getOutputGeometry();
                int nbrTextBoxes = OutputGeometry_.height;
                int nCol = OutputGeometry_.width;

                std::vector<NetOutput> tmp;
                // the output bounding box needs to be resized by the input height and width
                Size inputImageShape = Size(inputImage.cols(),inputImage.rows());
                NetOutput::getOutputs((float*)(netOutput.data),nbrTextBoxes,nCol,tmp,inputImageShape);
                // put the output in CV_OUT

                for (int k=0;k<nbrTextBoxes;k++)
                {
                    Bbox.push_back(tmp[k].bbox);
                    confidence.push_back(tmp[k].probability);
                }

     }



    void run(Mat& image, std::vector<Rect>* component_rects=NULL,
             std::vector<float>* component_confidences=NULL,
             int component_level=0)
    {
        CV_Assert(component_level==OCR_LEVEL_WORD);//Componnents not applicable for word spotting
        //double confidence;
        //String transcription;
        std::vector<Rect> bbox;
        std::vector<float> score;
        textDetectInImage(image,bbox,score);
        //output_text=transcription.c_str();
        if(component_rects!=NULL)
        {
            component_rects->resize(bbox.size());  // should be a user behavior

            component_rects = &bbox;
        }

        if(component_confidences!=NULL)
        {
            component_confidences->resize(score.size()); // shoub be a user behavior

            component_confidences = &score;
        }
    }

    void run(Mat& image, Mat& mask, std::vector<Rect>* component_rects=NULL,
             std::vector<float>* component_confidences=NULL,
             int component_level=0)
    {
        CV_Assert(mask.cols==image.cols && mask.rows== image.rows);//Mask is ignored because the CNN operates on a full image
        this->run(image,component_rects,component_confidences,component_level);
    }



    Ptr<TextRegionDetector> getClassifier()
    {
        return this->classifier_;
    }
};

Ptr<textDetector> textDetector::create(Ptr<TextRegionDetector> classifierPtr)
{
    return Ptr<textDetector>(new textDetectImpl(classifierPtr));
}

Ptr<textDetector> textDetector::create(String modelArchFilename, String modelWeightsFilename)
{

// create a custom preprocessor with rawval
    Ptr<ImagePreprocessor> preprocessor=ImagePreprocessor::createImageCustomPreprocessor(255);
// set the mean for the preprocessor

    Mat textbox_mean(1,3,CV_8U);
    textbox_mean.at<uchar>(0,0)=104;
    textbox_mean.at<uchar>(0,1)=117;
    textbox_mean.at<uchar>(0,2)=123;
    preprocessor->set_mean(textbox_mean);
// create a pointer to text box detector(textDetector)
    Ptr<TextRegionDetector> classifierPtr(DeepCNNTextDetector::create(modelArchFilename,modelWeightsFilename,preprocessor,1));
    return Ptr<textDetector>(new textDetectImpl(classifierPtr));
}







}  } //namespace text namespace cv
