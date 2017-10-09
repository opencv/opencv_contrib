#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include  "opencv2/highgui.hpp"
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

namespace cv { namespace text {
//************************************************************************************
//******************   ImagePreprocessor   *******************************************
//************************************************************************************

void ImagePreprocessor::preprocess(InputArray input,OutputArray output,Size sz,int outputChannels){
    Mat inpImg=input.getMat();
    Mat outImg;
    this->preprocess_(inpImg,outImg,sz,outputChannels);
    outImg.copyTo(output);
}
void ImagePreprocessor::set_mean(Mat mean){


    this->set_mean_(mean);

}



class ResizerPreprocessor: public ImagePreprocessor{
protected:
    void preprocess_(const Mat& input,Mat& output,Size outputSize,int outputChannels){
        //TODO put all the logic of channel and depth conversions in ImageProcessor class
        CV_Assert(outputChannels==1 || outputChannels==3);
        CV_Assert(input.channels()==1 || input.channels()==3);
        if(input.channels()!=outputChannels)
        {
            Mat tmpInput;
            if(outputChannels==1){
                cvtColor(input,tmpInput,COLOR_BGR2GRAY);
                if(input.depth()==CV_8U)
                {
                    tmpInput.convertTo(output,CV_32FC1,1/255.0);
                }else
                {//Assuming values are at the desired [0,1] range
                    tmpInput.convertTo(output, CV_32FC1);
                }
            }else
            {
                cvtColor(input,tmpInput,COLOR_GRAY2BGR);
                if(input.depth()==CV_8U)
                {
                    tmpInput.convertTo(output,CV_32FC3,1/255.0);
                }else
                {//Assuming values are at the desired [0,1] range
                    tmpInput.convertTo(output, CV_32FC3);
                }
            }
        }else
        {
            if(input.channels()==1)
            {
                if(input.depth()==CV_8U)
                {
                    input.convertTo(output, CV_32FC1,1/255.0);
                }else
                {//Assuming values are at the desired [0,1] range
                    input.convertTo(output, CV_32FC1);
                }
            }else
            {
                if(input.depth()==CV_8U){
                    input.convertTo(output, CV_32FC3,1/255.0);
                }else
                {//Assuming values are at the desired [0,1] range
                    input.convertTo(output, CV_32FC3);
                }
            }
        }
        if(outputSize.width!=0 && outputSize.height!=0)
        {
            resize(output,output,outputSize);
        }
    }
    //void set_mean_(Mat m){}
public:
    ResizerPreprocessor(){}
    ~ResizerPreprocessor(){}
};

class StandarizerPreprocessor: public ImagePreprocessor{
protected:
    double sigma_;
    //void set_mean_(Mat M){}

    void preprocess_(const Mat& input,Mat& output,Size outputSize,int outputChannels){

        //TODO put all the logic of channel and depth conversions in ImageProcessor class
        CV_Assert(outputChannels==1 || outputChannels==3);
        CV_Assert(input.channels()==1 || input.channels()==3);
        if(input.channels()!=outputChannels)
        {
            Mat tmpInput;
            if(outputChannels==1)
            {
                cvtColor(input,tmpInput,COLOR_BGR2GRAY);
                if(input.depth()==CV_8U)
                {
                    tmpInput.convertTo(output,CV_32FC1,1/255.0);
                }else
                {//Assuming values are at the desired [0,1] range
                    tmpInput.convertTo(output, CV_32FC1);
                }
            }else
            {
                cvtColor(input,tmpInput,COLOR_GRAY2BGR);
                if(input.depth()==CV_8U)
                {
                    tmpInput.convertTo(output,CV_32FC3,1/255.0);
                }else
                {//Assuming values are at the desired [0,1] range
                    tmpInput.convertTo(output, CV_32FC3);
                }
            }
        }else
        {
            if(input.channels()==1)
            {
                if(input.depth()==CV_8U)
                {
                    input.convertTo(output, CV_32FC1,1/255.0);
                }else
                {//Assuming values are at the desired [0,1] range
                    input.convertTo(output, CV_32FC1);
                }
            }else
            {
                if(input.depth()==CV_8U)
                {
                    input.convertTo(output, CV_32FC3,1/255.0);
                }else
                {//Assuming values are at the desired [0,1] range
                    input.convertTo(output, CV_32FC3);
                }
            }
        }
        if(outputSize.width!=0 && outputSize.height!=0)
        {
            resize(output,output,outputSize);
        }

        Scalar mean,dev;
        meanStdDev(output,mean,dev);
        subtract(output,mean[0],output);
        divide(output,(dev[0]/sigma_),output);
    }
public:
    StandarizerPreprocessor(double sigma):sigma_(sigma){}
    ~StandarizerPreprocessor(){}

};

class customPreprocessor:public ImagePreprocessor{
protected:

    double rawval_;
    Mat mean_;
    String channel_order_;

    void set_mean_(Mat imMean_){

        imMean_.copyTo(this->mean_);


    }

    void set_raw_scale(int rawval){
        rawval_ = rawval;

    }
    void set_channels(String channel_order){
        channel_order_=channel_order;
    }


    void preprocess_(const Mat& input,Mat& output,Size outputSize,int outputChannels){
        //TODO put all the logic of channel and depth conversions in ImageProcessor class

        CV_Assert(outputChannels==1 || outputChannels==3);
        CV_Assert(input.channels()==1 || input.channels()==3);
        if(input.channels()!=outputChannels)
        {
            Mat tmpInput;
            if(outputChannels==1)
            {
                cvtColor(input,tmpInput,COLOR_BGR2GRAY);
                if(input.depth()==CV_8U)
                {
                    if (rawval_ == 1)
                        tmpInput.convertTo(output,CV_32FC3,1/255.0);
                    else
                        tmpInput.convertTo(output,CV_32FC1);
                }else
                {//Assuming values are at the desired [0,1] range
                    if (rawval_ ==1)
                        tmpInput.convertTo(output, CV_32FC1);
                    else
                        tmpInput.convertTo(output, CV_32FC1,rawval_);
                }
            }else
            {
                cvtColor(input,tmpInput,COLOR_GRAY2BGR);
                if(input.depth()==CV_8U)
                {
                    if (rawval_ == 1)
                        tmpInput.convertTo(output,CV_32FC3,1/255.0);
                    else
                        tmpInput.convertTo(output,CV_32FC1);
                }else
                {//Assuming values are at the desired [0,1] range
                    if (rawval_ ==1)
                        tmpInput.convertTo(output, CV_32FC1);
                    else
                        tmpInput.convertTo(output, CV_32FC1,rawval_);
                }
            }
        }else
        {
            if(input.channels()==1)
            {
                if(input.depth()==CV_8U)
                {
                    if (rawval_ == 1)
                        input.convertTo(output,CV_32FC1,1/255.0);
                    else
                        input.convertTo(output,CV_32FC1);
                }else
                {//Assuming values are at the desired [0,1] range
                    if (rawval_ ==1)
                        input.convertTo(output, CV_32FC1);
                    else
                        input.convertTo(output, CV_32FC1,rawval_);
                }
            }else
            {
                if(input.depth()==CV_8U)
                {
                    if (rawval_ == 1)
                        input.convertTo(output,CV_32FC3,1/255.0);
                    else
                        input.convertTo(output,CV_32FC3);
                }else
                {//Assuming values are at the desired [0,1] range
                    if (rawval_ ==1)
                        input.convertTo(output, CV_32FC3);
                    else
                        input.convertTo(output, CV_32FC3,rawval_);
                }
            }
        }
        if(outputSize.width!=0 && outputSize.height!=0)
        {
            resize(output,output,outputSize);
        }

        if (!this->mean_.empty()){

            Scalar mean_s(this->mean_.at<uchar>(0,0),this->mean_.at<uchar>(0,1),this->mean_.at<uchar>(0,2));
            subtract(output,mean_s,output);
        }
        else{
            Scalar mean_s;
            mean_s = mean(output);
            subtract(output,mean_s,output);
        }

    }

public:
    customPreprocessor( double rawval,String channel_order):rawval_(rawval),channel_order_(channel_order){}
    ~customPreprocessor(){}

};

class MeanSubtractorPreprocessor: public ImagePreprocessor{
protected:
    Mat mean_;
    //void set_mean_(Mat m){}
    void preprocess_(const Mat& input,Mat& output,Size outputSize,int outputChannels){
        //TODO put all the logic of channel and depth conversions in ImageProcessor class
        CV_Assert(this->mean_.cols==outputSize.width && this->mean_.rows ==outputSize.height);
        CV_Assert(outputChannels==1 || outputChannels==3);
        CV_Assert(input.channels()==1 || input.channels()==3);
        if(input.channels()!=outputChannels)
        {
            Mat tmpInput;
            if(outputChannels==1)
            {
                cvtColor(input,tmpInput,COLOR_BGR2GRAY);
                if(input.depth()==CV_8U)
                {
                    tmpInput.convertTo(output,CV_32FC1,1/255.0);
                }else
                {//Assuming values are at the desired [0,1] range
                    tmpInput.convertTo(output, CV_32FC1);
                }
            }else
            {
                cvtColor(input,tmpInput,COLOR_GRAY2BGR);
                if(input.depth()==CV_8U)
                {
                    tmpInput.convertTo(output,CV_32FC3,1/255.0);
                }else
                {//Assuming values are at the desired [0,1] range
                    tmpInput.convertTo(output, CV_32FC3);
                }
            }
        }else
        {
            if(input.channels()==1)
            {
                if(input.depth()==CV_8U)
                {
                    input.convertTo(output, CV_32FC1,1/255.0);
                }else
                {//Assuming values are at the desired [0,1] range
                    input.convertTo(output, CV_32FC1);
                }
            }else
            {
                if(input.depth()==CV_8U)
                {
                    input.convertTo(output, CV_32FC3,1/255.0);
                }else
                {//Assuming values are at the desired [0,1] range
                    input.convertTo(output, CV_32FC3);
                }
            }
        }
        if(outputSize.width!=0 && outputSize.height!=0)
        {
            resize(output,output,outputSize);
        }
        subtract(output,this->mean_,output);
    }
public:
    MeanSubtractorPreprocessor(Mat mean)
    {
        mean.copyTo(this->mean_);
    }

    ~MeanSubtractorPreprocessor(){}
};



Ptr<ImagePreprocessor> ImagePreprocessor::createResizer()
{
    return Ptr<ImagePreprocessor>(new ResizerPreprocessor);
}

Ptr<ImagePreprocessor> ImagePreprocessor::createImageStandarizer(double sigma)
{
    return Ptr<ImagePreprocessor>(new StandarizerPreprocessor(sigma));
}
Ptr<ImagePreprocessor> ImagePreprocessor::createImageCustomPreprocessor(double rawval,String channel_order)
{

    return Ptr<ImagePreprocessor>(new customPreprocessor(rawval,channel_order));
}

Ptr<ImagePreprocessor> ImagePreprocessor::createImageMeanSubtractor(InputArray meanImg)
{
    Mat tmp=meanImg.getMat();
    return Ptr<ImagePreprocessor>(new MeanSubtractorPreprocessor(tmp));
}
}
}
