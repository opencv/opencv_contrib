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


#ifdef HAVE_CAFFE
#include "caffe/caffe.hpp"
#endif

namespace cv { namespace text {

//Maybe OpenCV has a routine better suited
inline bool fileExists (String filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

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
}


class DeepCNNCaffeImpl: public DeepCNN{
protected:
    void classifyMiniBatch(std::vector<Mat> inputImageList, Mat outputMat)
    {
        //Classifies a list of images containing at most minibatchSz_ images
        CV_Assert(int(inputImageList.size())<=this->minibatchSz_);
        CV_Assert(outputMat.isContinuous());

#ifdef HAVE_CAFFE
        net_->input_blobs()[0]->Reshape(inputImageList.size(), this->channelCount_,this->inputGeometry_.height,this->inputGeometry_.width);
        net_->Reshape();
        float* inputBuffer=net_->input_blobs()[0]->mutable_cpu_data();
        float* inputData=inputBuffer;

        for(size_t imgNum=0;imgNum<inputImageList.size();imgNum++)
        {
            std::vector<Mat> input_channels;
            Mat preprocessed;
            // if the image have multiple color channels the input layer should be populated accordingly
            for (int channel=0;channel < this->channelCount_;channel++){

                cv::Mat netInputWraped(this->inputGeometry_.height, this->inputGeometry_.width, CV_32FC1, inputData);
                input_channels.push_back(netInputWraped);
                //input_data += width * height;
                inputData+=(this->inputGeometry_.height*this->inputGeometry_.width);
            }
            this->preprocess(inputImageList[imgNum],preprocessed);
            split(preprocessed, input_channels);

        }
        this->net_->ForwardPrefilled();
        const float* outputNetData=net_->output_blobs()[0]->cpu_data();
        this->outputGeometry_ = Size(net_->output_blobs()[0]->width(),net_->output_blobs()[0]->height());
        int outputSz = this->outputSize_ * this->outputGeometry_.height * this->outputGeometry_.width;

        //outputMat.resize(this->outputGeometry_.height * this->outputGeometry_.width);
        float*outputMatData=(float*)(outputMat.data);
        memcpy(outputMatData,outputNetData,sizeof(float)*outputSz*inputImageList.size());

#endif
    }

//    void process_(Mat inputImage, Mat &outputMat)
//    {
//        // do forward pass and stores the output in outputMat
//        //Process one image
//        CV_Assert(this->minibatchSz_==1);
//        //CV_Assert(outputMat.isContinuous());

//#ifdef HAVE_CAFFE
//        net_->input_blobs()[0]->Reshape(1, this->channelCount_,this->inputGeometry_.height,this->inputGeometry_.width);
//        net_->Reshape();
//        float* inputBuffer=net_->input_blobs()[0]->mutable_cpu_data();
//        float* inputData=inputBuffer;

//        std::vector<Mat> input_channels;
//        Mat preprocessed;
//        // if the image have multiple color channels the input layer should be populated accordingly
//        for (int channel=0;channel < this->channelCount_;channel++){

//            cv::Mat netInputWraped(this->inputGeometry_.height, this->inputGeometry_.width, CV_32FC1, inputData);
//            input_channels.push_back(netInputWraped);
//            //input_data += width * height;
//            inputData+=(this->inputGeometry_.height*this->inputGeometry_.width);
//        }
//        this->preprocess(inputImage,preprocessed);
//        split(preprocessed, input_channels);

//        //preprocessed.copyTo(netInputWraped);


//        this->net_->Forward();
//        const float* outputNetData=net_->output_blobs()[0]->cpu_data();
//        // const float* outputNetData1=net_->output_blobs()[1]->cpu_data();




//        this->outputGeometry_ = Size(net_->output_blobs()[0]->width(),net_->output_blobs()[0]->height());
//        int outputSz = this->outputSize_ * this->outputGeometry_.height * this->outputGeometry_.width;
//        outputMat.create(this->outputGeometry_.height , this->outputGeometry_.width,CV_32FC1);
//        float*outputMatData=(float*)(outputMat.data);

//        memcpy(outputMatData,outputNetData,sizeof(float)*outputSz);



//#endif
//    }



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
        this->outputGeometry_=dn.outputGeometry_;
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
        this->channelCount_ = inputLayer->channels();

        inputLayer->Reshape(this->minibatchSz_,this->channelCount_,this->inputGeometry_.height, this->inputGeometry_.width);
        net_->Reshape();
        this->outputSize_=net_->output_blobs()[0]->channels();
        this->outputGeometry_ = Size(net_->output_blobs()[0]->width(),net_->output_blobs()[0]->height());





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
//    void detect(InputArray image, OutputArray Bbox_prob)
//    {

//        Bbox_prob.create(this->outputGeometry_,CV_32F); // dummy initialization is it needed
//        Mat outputMat = Bbox_prob.getMat();
//        process_(image.getMat(),outputMat);
//        //copy back to outputArray
//        outputMat.copyTo(Bbox_prob);
//    }

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
    Size getOutputGeometry()
    {
        return this->outputGeometry_;
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

class OCRHolisticWordRecognizerImpl: public OCRHolisticWordRecognizer{
private:
    struct NetOutput{
        //Auxiliary structure that handles the logic of getting class ids and probabillities from
        //the raw outputs of caffe
        int wordIdx;
        float probabillity;

        static bool sorter(const NetOutput& o1,const NetOutput& o2)
        {//used with std::sort to provide the most probable class
            return o1.probabillity>o2.probabillity;
        }

        static void getOutputs(const float* buffer,int nbOutputs,std::vector<NetOutput>& res)
        {
            res.resize(nbOutputs);
            for(int k=0;k<nbOutputs;k++)
            {
                res[k].wordIdx=k;
                res[k].probabillity=buffer[k];
            }
            std::sort(res.begin(),res.end(),NetOutput::sorter);
        }

        static void getClassification(const float* buffer,int nbOutputs,int &classNum,double& confidence)
        {
            std::vector<NetOutput> tmp;
            getOutputs(buffer,nbOutputs,tmp);
            classNum=tmp[0].wordIdx;
            confidence=tmp[0].probabillity;
        }
    };
protected:
    std::vector<String> labels_;
    Ptr<TextImageClassifier> classifier_;
public:
    OCRHolisticWordRecognizerImpl(Ptr<TextImageClassifier> classifierPtr,String vocabularyFilename):classifier_(classifierPtr)
    {
        CV_Assert(fileExists(vocabularyFilename));//this fails for some rason
        std::ifstream labelsFile(vocabularyFilename.c_str());
        if(!labelsFile)
        {
            CV_Error(Error::StsError,"Could not read Labels from file");
        }
        std::string line;
        while (std::getline(labelsFile, line))
        {
            labels_.push_back(std::string(line));
        }
        CV_Assert(this->classifier_->getOutputSize()==int(this->labels_.size()));
    }

    OCRHolisticWordRecognizerImpl(Ptr<TextImageClassifier> classifierPtr,const std::vector<String>& vocabulary):classifier_(classifierPtr)
    {
        this->labels_=vocabulary;
        CV_Assert(this->classifier_->getOutputSize()==int(this->labels_.size()));
    }

    void recogniseImage(InputArray inputImage,CV_OUT String& transcription,CV_OUT double& confidence)
    {
        Mat netOutput;
        this->classifier_->classify(inputImage,netOutput);
        int classNum;
        NetOutput::getClassification((float*)(netOutput.data),this->classifier_->getOutputSize(),classNum,confidence);
        transcription=this->labels_[classNum];
    }

    void recogniseImageBatch(InputArrayOfArrays inputImageList,CV_OUT std::vector<String>& transcriptionVec,CV_OUT std::vector<double>& confidenceVec)
    {
        Mat netOutput;
        this->classifier_->classifyBatch(inputImageList,netOutput);
        for(int k=0;k<netOutput.rows;k++)
        {
            int classNum;
            double confidence;
            NetOutput::getClassification((float*)(netOutput.row(k).data),this->classifier_->getOutputSize(),classNum,confidence);
            transcriptionVec.push_back(this->labels_[classNum]);
            confidenceVec.push_back(confidence);
        }
    }


    void run(Mat& image, std::string& output_text, std::vector<Rect>* component_rects=NULL,
             std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
             int component_level=0)
    {
        CV_Assert(component_level==OCR_LEVEL_WORD);//Componnents not applicable for word spotting
        double confidence;
        String transcription;
        recogniseImage(image,transcription,confidence);
        output_text=transcription.c_str();
        if(component_rects!=NULL)
        {
            component_rects->resize(1);
            (*component_rects)[0]=Rect(0,0,image.size().width,image.size().height);
        }
        if(component_texts!=NULL)
        {
            component_texts->resize(1);
            (*component_texts)[0]=transcription.c_str();
        }
        if(component_confidences!=NULL)
        {
            component_confidences->resize(1);
            (*component_confidences)[0]=float(confidence);
        }
    }

    void run(Mat& image, Mat& mask, std::string& output_text, std::vector<Rect>* component_rects=NULL,
             std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
             int component_level=0)
    {
        CV_Assert(mask.cols==image.cols && mask.rows== image.rows);//Mask is ignored because the CNN operates on a full image
        this->run(image,output_text,component_rects,component_texts,component_confidences,component_level);
    }

    std::vector<String>& getVocabulary()
    {
        return this->labels_;
    }

    Ptr<TextImageClassifier> getClassifier()
    {
        return this->classifier_;
    }
};

Ptr<OCRHolisticWordRecognizer> OCRHolisticWordRecognizer::create(Ptr<TextImageClassifier> classifierPtr,String vocabularyFilename )
{
    return Ptr<OCRHolisticWordRecognizer>(new OCRHolisticWordRecognizerImpl(classifierPtr,vocabularyFilename));
}

Ptr<OCRHolisticWordRecognizer> OCRHolisticWordRecognizer::create(String modelArchFilename, String modelWeightsFilename, String vocabularyFilename)
{
    Ptr<ImagePreprocessor> preprocessor=ImagePreprocessor::createImageStandarizer(113);
    Ptr<TextImageClassifier> classifierPtr(new DeepCNNCaffeImpl(modelArchFilename,modelWeightsFilename,preprocessor,100));
    return Ptr<OCRHolisticWordRecognizer>(new OCRHolisticWordRecognizerImpl(classifierPtr,vocabularyFilename));
}

Ptr<OCRHolisticWordRecognizer> OCRHolisticWordRecognizer::create(Ptr<TextImageClassifier> classifierPtr,const std::vector<String>& vocabulary)
{
    return Ptr<OCRHolisticWordRecognizer>(new OCRHolisticWordRecognizerImpl(classifierPtr,vocabulary));
}

Ptr<OCRHolisticWordRecognizer> OCRHolisticWordRecognizer::create(String modelArchFilename, String modelWeightsFilename,const std::vector<String>& vocabulary){
    Ptr<ImagePreprocessor> preprocessor=ImagePreprocessor::createImageStandarizer(113);
    Ptr<TextImageClassifier> classifierPtr(new DeepCNNCaffeImpl(modelArchFilename,modelWeightsFilename,preprocessor,100));
    return Ptr<OCRHolisticWordRecognizer>(new OCRHolisticWordRecognizerImpl(classifierPtr,vocabulary));
}





}  } //namespace text namespace cv
