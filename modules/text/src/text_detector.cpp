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

//Maybe OpenCV has a routine better suited
//inline bool fileExists (String filename) {
//    std::ifstream f(filename.c_str());
//    return f.good();
//}

//************************************************************************************
//******************   ImagePreprocessor   *******************************************
//************************************************************************************

/*void ImagePreprocessor::preprocess(InputArray input,OutputArray output,Size sz,int outputChannels){
    Mat inpImg=input.getMat();
    Mat outImg;
    this->preprocess_(inpImg,outImg,sz,outputChannels);
    outImg.copyTo(output);
}*/


/*class ResizerPreprocessor: public ImagePreprocessor{
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
public:
    ResizerPreprocessor(){}
    ~ResizerPreprocessor(){}
};

class StandarizerPreprocessor: public ImagePreprocessor{
protected:
    double sigma_;
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
        Scalar dev,mean;
        meanStdDev(output,mean,dev);
        subtract(output,mean[0],output);
        divide(output,(dev[0]/sigma_),output);
    }
public:
    StandarizerPreprocessor(double sigma):sigma_(sigma){}
    ~StandarizerPreprocessor(){}
};

class MeanSubtractorPreprocessor: public ImagePreprocessor{
protected:
    Mat mean_;
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

Ptr<ImagePreprocessor> ImagePreprocessor::createImageMeanSubtractor(InputArray meanImg)
{
    Mat tmp=meanImg.getMat();
    return Ptr<ImagePreprocessor>(new MeanSubtractorPreprocessor(tmp));
}

//************************************************************************************
//******************   TextImageClassifier   *****************************************
//************************************************************************************

void TextImageClassifier::preprocess(const Mat& input,Mat& output)
{
    this->preprocessor_->preprocess_(input,output,this->inputGeometry_,this->channelCount_);
}

void TextImageClassifier::setPreprocessor(Ptr<ImagePreprocessor> ptr)
{
    CV_Assert(!ptr.empty());
    preprocessor_=ptr;
}

Ptr<ImagePreprocessor> TextImageClassifier::getPreprocessor()
{
    return preprocessor_;
}*/

/*
class DeepCNNCaffeImpl: public DeepCNN{
protected:
    void classifyMiniBatch(std::vector<Mat> inputImageList, Mat outputMat)
    {
        //Classifies a list of images containing at most minibatchSz_ images
        CV_Assert(int(inputImageList.size())<=this->minibatchSz_);
        CV_Assert(outputMat.isContinuous());
#ifdef HAVE_CAFFE
        net_->input_blobs()[0]->Reshape(inputImageList.size(), 1,this->inputGeometry_.height,this->inputGeometry_.width);
        net_->Reshape();
        float* inputBuffer=net_->input_blobs()[0]->mutable_cpu_data();
        float* inputData=inputBuffer;
        for(size_t imgNum=0;imgNum<inputImageList.size();imgNum++)
        {
            Mat preprocessed;
            cv::Mat netInputWraped(this->inputGeometry_.height, this->inputGeometry_.width, CV_32FC1, inputData);
            this->preprocess(inputImageList[imgNum],preprocessed);
            preprocessed.copyTo(netInputWraped);
            inputData+=(this->inputGeometry_.height*this->inputGeometry_.width);
        }
        this->net_->ForwardPrefilled();
        const float* outputNetData=net_->output_blobs()[0]->cpu_data();
        float*outputMatData=(float*)(outputMat.data);
        memcpy(outputMatData,outputNetData,sizeof(float)*this->outputSize_*inputImageList.size());
#endif
    }

#ifdef HAVE_CAFFE
    Ptr<caffe::Net<float> > net_;
#endif
    //Size inputGeometry_;
    int minibatchSz_;//The existence of the assignment operator mandates this to be nonconst
    int outputSize_;
public:
    DeepCNNCaffeImpl(const DeepCNNCaffeImpl& dn):
        minibatchSz_(dn.minibatchSz_),outputSize_(dn.outputSize_){
        channelCount_=dn.channelCount_;
        inputGeometry_=dn.inputGeometry_;
        //Implemented to supress Visual Studio warning "assignment operator could not be generated"
#ifdef HAVE_CAFFE
        this->net_=dn.net_;
#endif
    }
    DeepCNNCaffeImpl& operator=(const DeepCNNCaffeImpl &dn)
    {
#ifdef HAVE_CAFFE
        this->net_=dn.net_;
#endif
        this->setPreprocessor(dn.preprocessor_);
        this->inputGeometry_=dn.inputGeometry_;
        this->channelCount_=dn.channelCount_;
        this->minibatchSz_=dn.minibatchSz_;
        this->outputSize_=dn.outputSize_;
        this->preprocessor_=dn.preprocessor_;
        return *this;
        //Implemented to supress Visual Studio warning "assignment operator could not be generated"
    }

    DeepCNNCaffeImpl(String modelArchFilename, String modelWeightsFilename,Ptr<ImagePreprocessor> preprocessor, int maxMinibatchSz)
        :minibatchSz_(maxMinibatchSz)
    {
        CV_Assert(this->minibatchSz_>0);
        CV_Assert(fileExists(modelArchFilename));
        CV_Assert(fileExists(modelWeightsFilename));
        CV_Assert(!preprocessor.empty());
        this->setPreprocessor(preprocessor);
#ifdef HAVE_CAFFE
        this->net_.reset(new caffe::Net<float>(modelArchFilename, caffe::TEST));
        CV_Assert(net_->num_inputs()==1);
        CV_Assert(net_->num_outputs()==1);
        CV_Assert(this->net_->input_blobs()[0]->channels()==1
                ||this->net_->input_blobs()[0]->channels()==3);
        this->channelCount_=this->net_->input_blobs()[0]->channels();
        this->net_->CopyTrainedLayersFrom(modelWeightsFilename);
        caffe::Blob<float>* inputLayer = this->net_->input_blobs()[0];
        this->inputGeometry_=Size(inputLayer->width(), inputLayer->height());
        inputLayer->Reshape(this->minibatchSz_,1,this->inputGeometry_.height, this->inputGeometry_.width);
        net_->Reshape();
        this->outputSize_=net_->output_blobs()[0]->channels();

#else
        CV_Error(Error::StsError,"Caffe not available during compilation!");
#endif
    }

    void classify(InputArray image, OutputArray classProbabilities)
    {
        std::vector<Mat> inputImageList;
        inputImageList.push_back(image.getMat());
        classifyBatch(inputImageList,classProbabilities);
    }

    void classifyBatch(InputArrayOfArrays inputImageList, OutputArray classProbabilities)
    {
        std::vector<Mat> allImageVector;
        inputImageList.getMatVector(allImageVector);
        size_t outputSize=size_t(this->outputSize_);//temporary variable to avoid int to size_t arithmentic
        size_t minibatchSize=size_t(this->minibatchSz_);//temporary variable to avoid int to size_t arithmentic
        classProbabilities.create(Size(int(outputSize),int(allImageVector.size())),CV_32F);
        Mat outputMat = classProbabilities.getMat();
        for(size_t imgNum=0;imgNum<allImageVector.size();imgNum+=minibatchSize)
        {
            size_t rangeEnd=imgNum+std::min<size_t>(allImageVector.size()-imgNum,minibatchSize);
            std::vector<Mat>::const_iterator from=std::vector<Mat>::const_iterator(allImageVector.begin()+imgNum);
            std::vector<Mat>::const_iterator to=std::vector<Mat>::const_iterator(allImageVector.begin()+rangeEnd);
            std::vector<Mat> minibatchInput(from,to);
            classifyMiniBatch(minibatchInput,outputMat.rowRange(int(imgNum),int(rangeEnd)));
        }
    }

    int getOutputSize()
    {
        return this->outputSize_;
    }

    int getMinibatchSize()
    {
        return this->minibatchSz_;
    }

    int getBackend()
    {
        return OCR_HOLISTIC_BACKEND_CAFFE;
    }
};


Ptr<DeepCNN> DeepCNN::create(String archFilename,String weightsFilename,Ptr<ImagePreprocessor> preprocessor,int minibatchSz,int backEnd)
{
    if(preprocessor.empty())
    {
        preprocessor=ImagePreprocessor::createResizer();
    }
    switch(backEnd){
    case OCR_HOLISTIC_BACKEND_CAFFE:
        return Ptr<DeepCNN>(new DeepCNNCaffeImpl(archFilename, weightsFilename,preprocessor, minibatchSz));
        break;
    case OCR_HOLISTIC_BACKEND_NONE:
    default:
        CV_Error(Error::StsError,"DeepCNN::create backend not implemented");
        return Ptr<DeepCNN>();
        break;
    }
}


Ptr<DeepCNN> DeepCNN::createDictNet(String archFilename,String weightsFilename,int backEnd)
{
    Ptr<ImagePreprocessor> preprocessor=ImagePreprocessor::createImageStandarizer(113);
    switch(backEnd){
    case OCR_HOLISTIC_BACKEND_CAFFE:
        return Ptr<DeepCNN>(new DeepCNNCaffeImpl(archFilename, weightsFilename,preprocessor, 100));
        break;
    case OCR_HOLISTIC_BACKEND_NONE:
    default:
        CV_Error(Error::StsError,"DeepCNN::create backend not implemented");
        return Ptr<DeepCNN>();
        break;
    }
}

namespace cnn_config{
namespace caffe_backend{

#ifdef HAVE_CAFFE

bool getCaffeGpuMode()
{
    return caffe::Caffe::mode()==caffe::Caffe::GPU;
}

void setCaffeGpuMode(bool useGpu)
{
    if(useGpu)
    {
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
    }else
    {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
    }
}

bool getCaffeAvailable()
{
    return true;
}

#else

bool getCaffeGpuMode()
{
    CV_Error(Error::StsError,"Caffe not available during compilation!");
    return 0;
}

void setCaffeGpuMode(bool useGpu)
{
    CV_Error(Error::StsError,"Caffe not available during compilation!");
    CV_Assert(useGpu==1);//Compilation directives force
}

bool getCaffeAvailable(){
    return 0;
}

#endif

}//namespace caffe
}//namespace cnn_config
*/

class textDetectImpl: public textDetector{
private:
    struct NetOutput{
        //Auxiliary structure that handles the logic of getting bounding box and confidences of textness from
        //the raw outputs of caffe
        Rect bbox;
        float probability;

//        static bool sorter(const NetOutput& o1,const NetOutput& o2)
//        {//used with std::sort to provide the most probable class
//            return o1.probabillity>o2.probabillity;
//        }

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
               // printf("%f %f %f %f\n",buffer[k*nCol+3],buffer[k*nCol+4],buffer[k*nCol+5],buffer[k*nCol+6]);
                res[k].probability=buffer[k*nCol+2];
            }
//            std::sort(res.begin(),res.end(),NetOutput::sorter);
        }

//        static void getDetections(const float* buffer,int nbOutputs,int &classNum,double& confidence)
//        {
//            std::vector<NetOutput> tmp;
//            getOutputs(buffer,nbOutputs,tmp);
//            classNum=tmp[0].wordIdx;
//            confidence=tmp[0].probabillity;
//        }
    };
protected:
    //std::vector<String> labels_;
    Ptr<TextImageClassifier> classifier_;
public:
    textDetectImpl(Ptr<TextImageClassifier> classifierPtr):classifier_(classifierPtr)
    {

    }



    void textDetectInImage(InputArray inputImage,CV_OUT std::vector<Rect>& Bbox,CV_OUT std::vector<float>& confidence)
    {
                Mat netOutput;
                //std::cout<<"started detect"<<std::endl;
                this->classifier_->detect(inputImage,netOutput);
                //std::cout<<"After Detect"<<std::endl;
                Size OutputGeometry_ = this->classifier_->getOutputGeometry();
                int nbrTextBoxes = OutputGeometry_.height;
                int nCol = OutputGeometry_.width;
                //std::cout<<nbrTextBoxes<<std::endl;
                std::vector<NetOutput> tmp;
                Size inputImageShape = Size(inputImage.cols(),inputImage.rows());
                NetOutput::getOutputs((float*)(netOutput.data),nbrTextBoxes,nCol,tmp,inputImageShape);
                //Bbox.resize(nbrTextBoxes);
                //confidence.resize(nbrTextBoxes);
                for (int k=0;k<nbrTextBoxes;k++)
                {
                    Bbox.push_back(tmp[k].bbox);
                    confidence.push_back(tmp[k].probability);
                }
                //Bbox = netOutput.data;
                //confidence = netOutput.data;

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

//    std::vector<String>& getVocabulary()
//    {
//        return this->labels_;
//    }

    Ptr<TextImageClassifier> getClassifier()
    {
        return this->classifier_;
    }
};

Ptr<textDetector> textDetector::create(Ptr<TextImageClassifier> classifierPtr)
{
    return Ptr<textDetector>(new textDetectImpl(classifierPtr));
}

Ptr<textDetector> textDetector::create(String modelArchFilename, String modelWeightsFilename)
{


    Ptr<ImagePreprocessor> preprocessor=ImagePreprocessor::createImageCustomPreprocessor(255);

    Mat textbox_mean(1,3,CV_8U);
    textbox_mean.at<uchar>(0,0)=104;
    textbox_mean.at<uchar>(0,1)=117;
    textbox_mean.at<uchar>(0,2)=123;
    preprocessor->set_mean(textbox_mean);

    Ptr<TextImageClassifier> classifierPtr(DeepCNN::create(modelArchFilename,modelWeightsFilename,preprocessor,1));
    return Ptr<textDetector>(new textDetectImpl(classifierPtr));
}







}  } //namespace text namespace cv
