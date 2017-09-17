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

#ifdef HAVE_DNN
#include "opencv2/dnn.hpp"
#endif

using namespace cv;
using namespace cv::dnn;
using namespace std;
namespace cv { namespace text {

//Maybe OpenCV has a routine better suited
inline bool fileExists (String filename) {
    std::ifstream f(filename.c_str());
    return f.good();
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

#ifdef HAVE_CAFFE
    Ptr<caffe::Net<float> > net_;
#endif
    //Size inputGeometry_;//=Size(100,32);
    int minibatchSz_;//The existence of the assignment operator mandates this to be nonconst
    int outputSize_;
    //Size outputGeometry_;
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

class DeepCNNOpenCvDNNImpl: public DeepCNN{
protected:

    void classifyMiniBatch(std::vector<Mat> inputImageList, Mat outputMat)
    {
        //Classifies a list of images containing at most minibatchSz_ images
        CV_Assert(int(inputImageList.size())<=this->minibatchSz_);
        CV_Assert(outputMat.isContinuous());

#ifdef HAVE_DNN

        std::vector<Mat> preProcessedImList; // to store preprocessed images, should it be handled inside preprocessing class?

        Mat preprocessed;
        // preprocesses each image in the inputImageList and push to preprocessedImList
        for(size_t imgNum=0;imgNum<inputImageList.size();imgNum++)
        {
            this->preprocess(inputImageList[imgNum],preprocessed);
            preProcessedImList.push_back(preprocessed);
        }
        // set input data blob in dnn::net
        net_->setInput(blobFromImages(preProcessedImList,1, this->inputGeometry_), "data");

        float*outputMatData=(float*)(outputMat.data);
       //Mat outputNet(inputImageList.size(),this->outputSize_,CV_32FC1,outputMatData) ;
       Mat outputNet = this->net_->forward();
       outputNet = outputNet.reshape(1, 1);

       float*outputNetData=(float*)(outputNet.data);

       memcpy(outputMatData,outputNetData,sizeof(float)*this->outputSize_*inputImageList.size());

#endif
    }

#ifdef HAVE_DNN
    Ptr<Net> net_;
#endif
    // hard coding input image size. anything in DNN library to get that from prototxt??
   // Size inputGeometry_;//=Size(100,32);
    int minibatchSz_;//The existence of the assignment operator mandates this to be nonconst
    int outputSize_;
    //Size outputGeometry_;//= Size(1,1);
    //int channelCount_;
   // int inputChannel_ ;//=1;
    int _inputHeight;
    int _inputWidth ;
    int _inputChannel ;
public:
    DeepCNNOpenCvDNNImpl(const DeepCNNOpenCvDNNImpl& dn):
        minibatchSz_(dn.minibatchSz_),outputSize_(dn.outputSize_){
        channelCount_=dn.channelCount_;
        inputGeometry_=dn.inputGeometry_;
        //Implemented to supress Visual Studio warning "assignment operator could not be generated"
#ifdef HAVE_DNN
        this->net_=dn.net_;
#endif
    }
    DeepCNNOpenCvDNNImpl& operator=(const DeepCNNOpenCvDNNImpl &dn)
    {
#ifdef HAVE_DNN
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

    DeepCNNOpenCvDNNImpl(String modelArchFilename, String modelWeightsFilename,Ptr<ImagePreprocessor> preprocessor, int maxMinibatchSz,int inputWidth =100,int inputHeight = 32,int inputChannel =1)
        :minibatchSz_(maxMinibatchSz),_inputWidth(inputWidth),_inputHeight(inputHeight),_inputChannel(inputChannel)
    {

        CV_Assert(this->minibatchSz_>0);
        CV_Assert(fileExists(modelArchFilename));
        CV_Assert(fileExists(modelWeightsFilename));
        CV_Assert(!preprocessor.empty());
        this->setPreprocessor(preprocessor);
#ifdef HAVE_DNN

        this->net_ = makePtr<Net>(readNetFromCaffe(modelArchFilename,modelWeightsFilename));



        if (this->net_.empty())
        {
            std::cerr << "Can't load network by using the following files: " << std::endl;
            std::cerr << "prototxt:   " << modelArchFilename << std::endl;
            std::cerr << "caffemodel: " << modelWeightsFilename << std::endl;
            //std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
            //std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
            exit(-1);
        }


        this->inputGeometry_=Size(_inputWidth,_inputHeight);// Size(inputLayer->width(), inputLayer->height());
        this->channelCount_ = _inputChannel;//inputLayer->channels();

        //inputLayer->Reshape(this->minibatchSz_,this->channelCount_,this->inputGeometry_.height, this->inputGeometry_.width);
        Ptr< Layer > outLayer=	net_->getLayer (net_->getLayerId (net_->getLayerNames()[net_->getLayerNames().size()-2]));
        //std::vector<Mat> blobs = outLayer->blobs;

        this->outputSize_=(outLayer->blobs)[1].size[0] ;//net_->output_blobs()[0]->channels();
        //this->outputGeometry_ = Size(1,1);//Size(net_->output_blobs()[0]->width(),net_->output_blobs()[0]->height());






#else
        CV_Error(Error::StsError,"DNN module not available during compilation!");
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
        return OCR_HOLISTIC_BACKEND_DNN;
    }
};

Ptr<DeepCNN> DeepCNN::create(String archFilename,String weightsFilename,Ptr<ImagePreprocessor> preprocessor,int minibatchSz,int backEnd)
{
    if(preprocessor.empty())
    {
        preprocessor=ImagePreprocessor::createResizer();
    }
    switch(backEnd){
    case OCR_HOLISTIC_BACKEND_DEFAULT:

#ifdef HAVE_CAFFE
        return Ptr<DeepCNN>(new DeepCNNCaffeImpl(archFilename, weightsFilename,preprocessor, minibatchSz));

#elif defined(HAVE_DNN)
        return Ptr<DeepCNN>(new DeepCNNOpenCvDNNImpl(archFilename, weightsFilename,preprocessor, minibatchSz));
#else
        CV_Error(Error::StsError,"DeepCNN::create backend not implemented");
        return Ptr<DeepCNN>();
#endif
        break;

    case OCR_HOLISTIC_BACKEND_CAFFE:
        return Ptr<DeepCNN>(new DeepCNNCaffeImpl(archFilename, weightsFilename,preprocessor, minibatchSz));
        break;
  case OCR_HOLISTIC_BACKEND_DNN:
        return Ptr<DeepCNN>(new DeepCNNOpenCvDNNImpl(archFilename, weightsFilename,preprocessor, minibatchSz));
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
    case OCR_HOLISTIC_BACKEND_DEFAULT:

#ifdef HAVE_CAFFE
        return Ptr<DeepCNN>(new DeepCNNCaffeImpl(archFilename, weightsFilename,preprocessor, 100));

#elif defined(HAVE_DNN)
        return Ptr<DeepCNN>(new DeepCNNOpenCvDNNImpl(archFilename, weightsFilename,preprocessor, 100));
#else
        CV_Error(Error::StsError,"DeepCNN::create backend not implemented");
        return Ptr<DeepCNN>();
#endif
        break;

    case OCR_HOLISTIC_BACKEND_CAFFE:
        return Ptr<DeepCNN>(new DeepCNNCaffeImpl(archFilename, weightsFilename,preprocessor, 100));
        break;
   case OCR_HOLISTIC_BACKEND_DNN:
        return Ptr<DeepCNN>(new DeepCNNOpenCvDNNImpl(archFilename, weightsFilename,preprocessor, 100));
        break;
    case OCR_HOLISTIC_BACKEND_NONE:
    default:
        CV_Error(Error::StsError,"DeepCNN::create backend not implemented");
        return Ptr<DeepCNN>();
        break;
    }
}

namespace cnn_config{
std::vector<std::string> getAvailableBackends()
{
    std::vector<std::string> backends;

#ifdef HAVE_CAFFE
    backends.push_back("CAFFE, OCR_HOLISTIC_BACKEND_CAFFE"); // dnn backend opencv_dnn

#endif
#ifdef HAVE_DNN
    backends.push_back("DNN, OCR_HOLISTIC_BACKEND_DNN");// opencv_dnn based backend"
#endif
    return backends;


}

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
namespace dnn_backend{
#ifdef  HAVE_DNN


bool getDNNAvailable(){
    return true;
}
#else
bool getDNNAvailable(){
    return 0;
}
#endif
}//namspace dnn_backend
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
