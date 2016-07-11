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

//should this be moved elsewhere?
//In precomp.hpp It doesn't work
#ifdef HAVE_CAFFE
#include "caffe/caffe.hpp"
#endif


namespace cv { namespace text {

//Maybe OpenCV has a routine better suited
inline bool fileExists (String filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}


class DictNetCaffeImpl: public DictNet{
protected:
    void preprocess(Mat& input,Mat& output){
        if(input.channels()==3){
            Mat tmpInput;
            cvtColor(input,tmpInput,COLOR_BGR2GRAY);
            if(input.depth()==CV_8U){
                tmpInput.convertTo(output,CV_32FC1,1/255.0);
            }else{//Assuming values are at the desired [0,1] range
                tmpInput.convertTo(output, CV_32FC1);
            }
        }else if(input.channels()==1){
            if(input.depth()==CV_8U){
                input.convertTo(output, CV_32FC1,1/255.0);
            }else{//Assuming values are at the desired [0,1] range
                input.convertTo(output, CV_32FC1);
            }
        }else{
            CV_Error(Error::StsError,"Expecting images with either 1 or 3 channels");
        }
        resize(output,output,this->inputGeometry_);
        Scalar dev,mean;
        meanStdDev(output,mean,dev);
        subtract(output,mean[0],output);
        divide(output,(dev[0]/128.0),output);
    }

    void classifyMiniBatch(std::vector<Mat> inputImageList, Mat outputMat){
        //Classifies a list of images containing at most minibatchSz_ images
        CV_Assert(int(inputImageList.size())<=this->minibatchSz_);
        CV_Assert(outputMat.isContinuous());
#ifdef HAVE_CAFFE
        net_->input_blobs()[0]->Reshape(inputImageList.size(), 1,this->inputGeometry_.height,this->inputGeometry_.width);
        net_->Reshape();
        float* inputBuffer=net_->input_blobs()[0]->mutable_cpu_data();
        float* inputData=inputBuffer;
        for(size_t imgNum=0;imgNum<inputImageList.size();imgNum++){
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
    Size inputGeometry_;
    const int minibatchSz_;
    const bool gpuBackend_;
    int outputSize_;
public:
    DictNetCaffeImpl(const DictNetCaffeImpl& dn):inputGeometry_(dn.inputGeometry_),minibatchSz_(dn.minibatchSz_),
        gpuBackend_(dn.gpuBackend_),outputSize_(dn.outputSize_){
        //Implemented to supress Visual Studio warning
#ifdef HAVE_CAFFE
        this->net_=dn.net_;
#endif
    }

    DictNetCaffeImpl(String modelArchFilename, String modelWeightsFilename, int maxMinibatchSz, bool useGpu)
        :minibatchSz_(maxMinibatchSz), gpuBackend_(useGpu){
        CV_Assert(this->minibatchSz_>0);
        CV_Assert(fileExists(modelArchFilename));
        CV_Assert(fileExists(modelWeightsFilename));
#ifdef HAVE_CAFFE
        if(this->gpuBackend_){
            caffe::Caffe::set_mode(caffe::Caffe::GPU);
        }else{
            caffe::Caffe::set_mode(caffe::Caffe::CPU);
        }
        this->net_.reset(new caffe::Net<float>(modelArchFilename, caffe::TEST));
        CV_Assert(net_->num_inputs()==1);
        CV_Assert(net_->num_outputs()==1);
        CV_Assert(this->net_->input_blobs()[0]->channels()==1);
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

    void classify(InputArray image, OutputArray classProbabilities){
        std::vector<Mat> inputImageList;
        inputImageList.push_back(image.getMat());
        classifyBatch(inputImageList,classProbabilities);
    }

    void classifyBatch(InputArrayOfArrays inputImageList, OutputArray classProbabilities){
        std::vector<Mat> allImageVector;
        inputImageList.getMatVector(allImageVector);
        classProbabilities.create(Size((size_t)(this->outputSize_),allImageVector.size()),CV_32F);
        Mat outputMat = classProbabilities.getMat();
        for(size_t imgNum=0;imgNum<allImageVector.size();imgNum+=this->minibatchSz_){
            int rangeEnd=imgNum+std::min<int>(allImageVector.size()-imgNum,this->minibatchSz_);
            std::vector<Mat>::const_iterator from=allImageVector.begin()+imgNum;
            std::vector<Mat>::const_iterator to=allImageVector.begin()+rangeEnd;
            std::vector<Mat> minibatchInput(from,to);
            classifyMiniBatch(minibatchInput,outputMat.rowRange(imgNum,rangeEnd));
        }
    }

    int getOutputSize(){
        return this->outputSize_;
    }
    int getMinibatchSize(){
        return this->minibatchSz_;
    }
    bool usingGpu(){
        return this->gpuBackend_;
    }
    int getBackend(){
        return OCR_HOLISTIC_BACKEND_CAFFE;
    }
};


Ptr<DictNet> DictNet::create(String archFilename,String weightsFilename,int minibatchSz,bool useGpu,int backEnd){
    switch(backEnd){
    case OCR_HOLISTIC_BACKEND_CAFFE:
        return Ptr<DictNet>(new DictNetCaffeImpl(archFilename, weightsFilename, minibatchSz, useGpu));
        break;
    case OCR_HOLISTIC_BACKEND_NONE:
    default:
        CV_Error(Error::StsError,"DictNet::create backend not implemented");
        return Ptr<DictNet>();
        break;
    }
}


class OCRHolisticWordRecognizerImpl: public OCRHolisticWordRecognizer{
private:
    struct NetOutput{
        //Auxiliary structure that handles the logic of getting class ids and probabillities from
        //the raw outputs of caffe
        int wordIdx;
        float probabillity;

        static bool sorter(const NetOutput& o1,const NetOutput& o2){//used with std::sort to provide the most probable class
            return o1.probabillity>o2.probabillity;
        }

        static void getOutputs(const float* buffer,int nbOutputs,std::vector<NetOutput>& res){
            res.resize(nbOutputs);
            for(int k=0;k<nbOutputs;k++){
                res[k].wordIdx=k;
                res[k].probabillity=buffer[k];
            }
            std::sort(res.begin(),res.end(),NetOutput::sorter);
        }
        static void getClassification(const float* buffer,int nbOutputs,int &classNum,double& confidence){
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
    OCRHolisticWordRecognizerImpl(Ptr<TextImageClassifier> classifierPtr,String vocabullaryFilename):classifier_(classifierPtr){
        CV_Assert(fileExists(vocabullaryFilename));//this fails for some rason
        std::ifstream labelsFile(vocabullaryFilename.c_str());
        if(!labelsFile){
            CV_Error(Error::StsError,"Could not read Labels from file");
        }
        std::string line;
        while (std::getline(labelsFile, line)){
            labels_.push_back(std::string(line));
        }
        CV_Assert(this->classifier_->getOutputSize()==int(this->labels_.size()));
    }

    void recogniseImage(InputArray inputImage,CV_OUT String& transcription,CV_OUT double& confidence){
        Mat netOutput;
        this->classifier_->classify(inputImage,netOutput);
        int classNum;
        NetOutput::getClassification((float*)(netOutput.data),this->classifier_->getOutputSize(),classNum,confidence);
        transcription=this->labels_[classNum];
    }
    void recogniseImageBatch(InputArrayOfArrays inputImageList,CV_OUT std::vector<String>& transcriptionVec,CV_OUT std::vector<double>& confidenceVec){
        Mat netOutput;
        this->classifier_->classifyBatch(inputImageList,netOutput);
        for(int k=0;k<netOutput.rows;k++){
            int classNum;
            double confidence;
            NetOutput::getClassification((float*)(netOutput.row(k).data),this->classifier_->getOutputSize(),classNum,confidence);
            transcriptionVec.push_back(this->labels_[classNum]);
            confidenceVec.push_back(confidence);
        }
    }


    void run(Mat& image, std::string& output_text, std::vector<Rect>* component_rects=NULL,
             std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
             int component_level=0){
        CV_Assert(component_level==OCR_LEVEL_WORD);//Componnents not applicable for word spotting
        double confidence;
        String transcription;
        recogniseImage(image,transcription,confidence);
        output_text=transcription.c_str();
        if(component_rects!=NULL){
            component_rects->resize(1);
            (*component_rects)[0]=Rect(0,0,image.size().width,image.size().height);
        }
        if(component_texts!=NULL){
            component_texts->resize(1);
            (*component_texts)[0]=transcription.c_str();
        }
        if(component_confidences!=NULL){
            component_confidences->resize(1);
            (*component_confidences)[0]=float(confidence);
        }
    }
    void run(Mat& image, Mat& mask, std::string& output_text, std::vector<Rect>* component_rects=NULL,
             std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
             int component_level=0){
        CV_Assert(mask.cols==image.cols && mask.rows== image.rows);//Mask is ignored because the CNN operates on a full image
        this->run(image,output_text,component_rects,component_texts,component_confidences,component_level);
    }
    std::vector<String>& getVocabulary(){
        return this->labels_;
    }
};

Ptr<OCRHolisticWordRecognizer> OCRHolisticWordRecognizer::create(Ptr<TextImageClassifier> classifierPtr,String vocabullaryFilename ){
    return Ptr<OCRHolisticWordRecognizer>(new OCRHolisticWordRecognizerImpl(classifierPtr,vocabullaryFilename));
}

Ptr<OCRHolisticWordRecognizer> OCRHolisticWordRecognizer::create(String modelArchFilename, String modelWeightsFilename, String vocabullaryFilename){
    Ptr<TextImageClassifier> classifierPtr(new DictNetCaffeImpl(modelArchFilename,modelWeightsFilename, 100,0));
    return Ptr<OCRHolisticWordRecognizer>(new OCRHolisticWordRecognizerImpl(classifierPtr,vocabullaryFilename));
}

}  } //namespace text namespace cv
