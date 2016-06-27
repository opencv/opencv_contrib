#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/highgui.hpp"

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


//#define HAVE_CAFFE
/* According to https://github.com/BVLC/caffe/issues/1761
 * protoc src/caffe/proto/caffe.proto --cpp_out=.
 * mkdir include/caffe/proto
 * mv src/caffe/proto/caffe.pb.h include/caffe/proto
 */

#ifdef HAVE_CAFFE
#include "caffe/caffe.hpp"
#endif


namespace cv { namespace text {

//using namespace std;

//using namespace caffe;

//Auxiliary macros to be removed at the end of development
int __DBGCOUNTER=0;
#define DBGIMG(img,show,name) if(show){namedWindow( name, WINDOW_AUTOSIZE );imshow( name, img );waitKey(100000);  }
#define DBG() if(1){std::cerr<<"F:"<<__FILE__<<" L:"<<__LINE__<<" C:"<<__DBGCOUNTER<<"\n";std::cerr.flush();__DBGCOUNTER++;}
#define DBG2(msg) if(1){std::cerr<<"F:"<<__FILE__<<" L:"<<__LINE__<<" C:"<<__DBGCOUNTER<<"\t"<<msg<<"\n";std::cerr.flush();__DBGCOUNTER++;}


inline bool fileExists (std::string filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}


void OCRDictnet::run(Mat& image, std::string& output_text, std::vector<Rect>* component_rects,
                 std::vector<std::string>* component_texts, std::vector<float>* component_confidences,
                 int component_level){
    std::stringstream f;
    f<<image.size().width<<"\n"<<output_text<<"\n"<<component_rects->size()<<"\n"<<component_texts->size()<<"\n"<<component_confidences->size()<<"\n"<<component_level<<"\n";
}

void OCRDictnet::run(Mat& image, Mat& mask, std::string& output_text, std::vector<Rect>* component_rects,
                 std::vector<std::string>* component_texts, std::vector<float>* component_confidences,
                 int component_level){
    std::stringstream f;
    f<<image.size().width<<mask.size().width<<"\n"<<output_text<<"\n"<<component_rects->size()<<"\n"<<component_texts->size()<<"\n"<<component_confidences->size()<<"\n"<<component_level<<"\n";
}


CV_WRAP String OCRDictnet::run(InputArray image, int min_confidence, int component_level)
{
    std::string output1;
    std::string output2;
    std::vector<std::string> component_texts;
    std::vector<float> component_confidences;
    Mat image_m = image.getMat();
    run(image_m, output1, NULL, &component_texts, &component_confidences, component_level);
    for(unsigned int i = 0; i < component_texts.size(); i++){
        if(component_confidences[i] > min_confidence){
            output2 += component_texts[i];
        }
    }
    return String(output2);
}

CV_WRAP String OCRDictnet::run(InputArray image, InputArray mask, int min_confidence, int component_level){
    std::string output1;
    std::string output2;
    std::vector<std::string> component_texts;
    std::vector<float> component_confidences;
    Mat image_m = image.getMat();
    Mat mask_m = mask.getMat();
    run(image_m, mask_m, output1, NULL, &component_texts, &component_confidences, component_level);
    for(unsigned int i = 0; i < component_texts.size(); i++){
        std::cout << "confidence: " << component_confidences[i] << " text:" << component_texts[i] << std::endl;

        if(component_confidences[i] > min_confidence){
            output2 += component_texts[i];
        }
    }
    return String(output2);
}

/* Auxiliary class used to store CNN outputs */
struct NetOutput{
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
};



class OCRDictnetCaffeImpl : public OCRDictnet
{
protected:

#ifdef HAVE_CAFFE
    Ptr<caffe::Net<float> > net_;
#endif
    Size inputGeometry_;
    int numChannels_;
    const int minibatchSz_;
    const bool gpuBackend_;
    Ptr<Mat> meanImage_;
    bool standarize_;
    std::vector<std::string> labels_;

public:
    //Default constructor
    OCRDictnetCaffeImpl(std::string modelArchFilename, std::string modelWeightsFilename, std::string labelFile,
                   std::string modelAverageFilename="", int maxMinibatchSz=1, bool useGpu=1,bool standarizePixels=1)
        :minibatchSz_(maxMinibatchSz), gpuBackend_(useGpu), standarize_(standarizePixels){
        CV_Assert(this->minibatchSz_>0);
        CV_Assert(fileExists(modelArchFilename));
        CV_Assert(fileExists(modelWeightsFilename));
        CV_Assert(fileExists(labelFile));
        CV_Assert(modelAverageFilename=="" || fileExists(modelAverageFilename));
        std::ifstream labelsFile(labelFile.c_str());
        if(!labelsFile){
            std::cerr<<"Could not read Labels";
            throw cv::Exception();
        }
        std::string line;
        while (std::getline(labelsFile, line)){
            labels_.push_back(std::string(line));
        }

#ifdef HAVE_CAFFE
        if(this->gpuBackend_){
            caffe::Caffe::set_mode(caffe::Caffe::GPU);
        }else{
            caffe::Caffe::set_mode(caffe::Caffe::CPU);
        }
        DBG();
        this->net_.reset(new caffe::Net<float>(modelArchFilename, caffe::TEST));
        DBG();
        CV_Assert(net_->num_inputs()==1);
        CV_Assert(net_->num_outputs()==1);
        this->net_->CopyTrainedLayersFrom(modelWeightsFilename);
        DBG();
        caffe::Blob<float>* inputLayer = this->net_->input_blobs()[0];
        DBG();
        this->numChannels_ = inputLayer->channels();
        DBG();
        this->inputGeometry_=Size(inputLayer->width(), inputLayer->height());
        /* update the minibatch sise */
        DBG();
        std::cerr<<"minibatchSz_:"<<minibatchSz_<<"\tnumChannels_:"<<numChannels_<<"\tGeometry:"<<this->inputGeometry_<<"\n";
        inputLayer->Reshape(this->minibatchSz_,this->numChannels_,this->inputGeometry_.height, this->inputGeometry_.width);
        DBG();
        net_->Reshape();
        //Blob<float>* output_layer = net_->output_blobs()[0];
        CV_Assert(labels_.size()== net_->output_blobs()[0]->channels());
#else
        std::cout<<"Caffe not available during compilation!"<<std::endl;
#endif
        CV_Assert(this->numChannels_==3 || this->numChannels_==1);
        CV_Assert(this->meanImage_.get()==NULL || (this->meanImage_->cols==this->inputGeometry_.width && this->meanImage_->rows==this->inputGeometry_.height));
        DBG();
    }

#ifdef HAVE_CAFFE
public:
    std::vector<String> classify(InputArrayOfArrays inputImages,OutputArray outProbabillities){
        DBG();
        std::vector<Mat> inputImageList;
        inputImages.getMatVector(inputImageList);
        std::vector<String> outCaptions(inputImageList.size());
        std::vector<std::vector<NetOutput > > outputs(inputImageList.size());
        for(int batchStart=0;batchStart<inputImageList.size();batchStart+=minibatchSz_){
            int batchSz=((inputImageList.size()-batchStart)>this->minibatchSz_)?(this->minibatchSz_):(inputImageList.size()-batchStart);
            std::vector<Mat> tmpMatVec(batchSz);
            for(int k=0;k<batchSz;k++){
                tmpMatVec[k]=inputImageList[batchStart+k];
            }
            classifyMinibatch(tmpMatVec,&(outputs[batchStart]));
        }
        DBG();
        outCaptions.resize(inputImageList.size());
        Mat outProbabillitiesMat=Mat(inputImageList.size(),1,CV_32FC1);
        for(int sampleCount=0;sampleCount<inputImageList.size();sampleCount++){
            outCaptions[sampleCount]=this->labels_[outputs[sampleCount][0].wordIdx];
            outProbabillitiesMat.at<float>(sampleCount,0)=outputs[sampleCount][0].probabillity;
        }
        DBG();
        outProbabillitiesMat.copyTo(outProbabillities);
        return outCaptions;
        //outProbabillities.resize(inputImageList.size());
    }

protected:

    void classifyMinibatch(const std::vector<Mat>& inputImageList , std::vector<NetOutput>  *output){
        int imagePixelCount=this->inputGeometry_.width*this->inputGeometry_.height*this->numChannels_;
        net_->input_blobs()[0]->Reshape(inputImageList.size(), this->numChannels_,this->inputGeometry_.height,this->inputGeometry_.width);
        net_->Reshape();
        float* inputBuffer=net_->input_blobs()[0]->mutable_cpu_data();
        float* inputData=inputBuffer;
        int imgNum;
        for(imgNum=0;imgNum<inputImageList.size();imgNum++){
            Mat preproced;
            DBG();
            this->preprocess(inputImageList[imgNum],preproced);
            DBG();
            std::vector<Mat> channels(this->numChannels_);
            split(preproced, channels);
            for(int channelNum=0;channelNum<this->numChannels_;channelNum++){
                cv::Mat channel(this->inputGeometry_.height, this->inputGeometry_.width, CV_32FC1, inputData);
                std::cerr<<"Img Num:"<<imgNum<<"\nChannel"<<channelNum<<"\n";
                channels[channelNum].copyTo(channel);
                inputData+=(this->inputGeometry_.height*this->inputGeometry_.width);
            }
        }
        this->net_->ForwardPrefilled();
        const float* outputData=net_->output_blobs()[0]->cpu_data();
        for(imgNum=0;imgNum<inputImageList.size();imgNum++){
            NetOutput::getOutputs(outputData,this->labels_.size(),output[imgNum]);
        }
    }


    void preprocess1Channel(const Mat& inputImg, Mat& out){
        Mat resized;
        Mat floatImage;
        Mat colorAdjusted;
        Mat centeredImage;
        resize(inputImg,resized,this->inputGeometry_);
        if(resized.channels()==3){
            resized.convertTo(floatImage,CV_32FC1);
            cvtColor(floatImage,colorAdjusted,COLOR_BGR2GRAY);
        }else{
            resized.convertTo(colorAdjusted,CV_32FC1);
        }
        if(this->meanImage_.get()==NULL){
            if(this->standarize_){
                Scalar mean,stdDev;
                meanStdDev(colorAdjusted, mean, stdDev );
                centeredImage = (colorAdjusted - mean[0]) / ((stdDev[0] + 0.0001)/128 );
            }else{
                centeredImage = colorAdjusted - mean(colorAdjusted)[0];
            }
        }else{
            if(this->standarize_){
                Scalar mean,stdDev;
                meanStdDev(colorAdjusted, mean, stdDev );
                centeredImage = (colorAdjusted - *meanImage_) / ((stdDev[0] + 0.0001)/128 );
            }else{
                centeredImage = colorAdjusted - *meanImage_;
            }
        }
        centeredImage.copyTo(out);
    }


    void preprocess(const Mat& inputImg, Mat& out){
        if(this->numChannels_==1 && this->meanImage_.get() == NULL){
            preprocess1Channel(inputImg,out);
            return;
        }
        //TODOImplement preprocessing for 3 channels
        std::cerr<<"NOT IMPLEMETED\n\n";
    }

    void loadMean(std::string meanFilename){
        //TODO FIX
        if(meanFilename.length()==0){
            this->meanImage_.release();
            return;
        }
        caffe::BlobProto blobProto;
        caffe::ReadProtoFromBinaryFileOrDie(meanFilename.c_str(), &blobProto);
        /* Convert from BlobProto to Blob<float> */
        caffe::Blob<float> meanBlob;
        meanBlob.FromProto(blobProto);
        CV_Assert(meanBlob.channels()== this->numChannels_);
        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<Mat> channels;
        float* data = meanBlob.mutable_cpu_data();
        for (int i = 0; i < this->numChannels_; ++i) {
          /* Extract an individual channel. */
          Mat channel(meanBlob.height(), meanBlob.width(), CV_32FC1, data);
          channels.push_back(channel);
          data += meanBlob.height() * meanBlob.width();
        }
        /* Merge the separate channels into a single image. */
        this->meanImage_.reset(new Mat());
        merge(channels, *meanImage_);
    }

public:
    //Trivial Getters And Setters
    void setMean(InputArray arr){
        Mat meanImg=arr.getMat();
        CV_Assert(meanImg.channels()==this->numChannels_);
        CV_Assert(meanImg.cols ==this->inputGeometry_.width && meanImg.rows ==this->inputGeometry_.height);
        if(this->numChannels_==1){
            meanImg.convertTo(*meanImage_,CV_32FC1);
        }else{//numChannels==3
            meanImg.convertTo(*meanImage_,CV_32FC3);
        }
    }

    void getMean(OutputArray out){
        if(this->meanImage_.get()!=NULL){
            if(this->numChannels_==1){
                out.create(this->inputGeometry_, CV_32FC1);
            }else{//this->numChannels_==3
                out.create(this->inputGeometry_, CV_32FC3);
            }
            this->meanImage_->copyTo(out.getMat());
        }
    }

    std::vector<String> getVocabulary(){
        std::vector<String> res(this->labels_.size());
        for(int k =0;k<labels_.size();k++){
            res[k]=labels_[k];
        }
        return res;
    }

    void setVocabulary(const std::vector<String>& voc){
        CV_Assert(voc.size()==this->labels_.size());
        for(int k =0;k<labels_.size();k++){
            labels_[k]=voc[k].c_str();
        }
    }

    Size getInputShape(){
        return this->inputGeometry_;
    }

    int getInputChannels(){
        return this->numChannels_;
    }

    ~OCRDictnetCaffeImpl(){

    }
    //Functions Mandated By BaseOCR. TODO
    void run(Mat& image, std::string& output, std::vector<Rect>* component_rects=NULL,
             std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
             int component_level=0){
    CV_Assert( (image.type() == CV_8UC1) || (image.type() == CV_8UC3) );
    std::stringstream f;
    f<<image.size().width<<output<<component_rects->size()<<component_texts->size()<<component_confidences->size()<<component_level;

    }

    void run(Mat& image, Mat& mask, std::string& output, std::vector<Rect>* component_rects=NULL,
             std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
             int component_level=0){
        CV_Assert( mask.type() == CV_8UC1 );
        CV_Assert( (image.type() == CV_8UC1) || (image.type() == CV_8UC3) );

        run( mask, output, component_rects, component_texts, component_confidences, component_level);
    }

#endif //if HAVE_CAFFE
};



//CV_WRAP String OCRDictnet::run(InputArray image, int min_confidence, int component_level=0){}

//CV_WRAP String OCRDictnet::run(InputArray image, InputArray mask, int min_confidence, int component_level=0){}



Ptr<OCRDictnet> OCRDictnet::create(String modelArchFilename, String modelWeightsFilename, String dictionaryFilename,
                                                  String modelAverageFilename, int minibatchSize,bool useGpu,bool standarizePixels){
    //TODO Branch backend engines, for now only caffe available
    return makePtr<OCRDictnetCaffeImpl>(modelArchFilename,modelWeightsFilename,dictionaryFilename,modelAverageFilename,minibatchSize,useGpu,standarizePixels);

}


}  } //namespace text namespace cv
