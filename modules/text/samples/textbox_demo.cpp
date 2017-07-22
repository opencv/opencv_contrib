/*
 * dictnet_demo.cpp
 *
 * Demonstrates simple use of the holistic word classifier in C++
 *
 * Created on: June 26, 2016
 *     Author: Anguelos Nicolaou <anguelos.nicolaou AT gmail.com>
 */

#include  "opencv2/text.hpp"
#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"

#include  <sstream>
#include  <vector>
#include  <iostream>
#include  <iomanip>
#include  <fstream>

void textbox_draw(cv::Mat &src, std::vector<cv::Rect>  &groups,std::vector<float> &probs,std::vector<cv::String> wordList,float thres);
inline std::string getHelpStr(std::string progFname){
    std::stringstream out;
    out << "    Demo of text detection CNN for text detection." << std::endl;
    out << "    Max Jaderberg et al.: Reading Text in the Wild with Convolutional Neural Networks, IJCV 2015"<<std::endl<<std::endl;

    out << "    Usage: " << progFname << " <output_file> <input_image>" << std::endl;
    out << "    Caffe Model files  (textbox.caffemodel, textbox_deploy.prototxt)"<<std::endl;
    out << "      must be in the current directory." << std::endl << std::endl;

    out << "    Obtaining Caffe Model files in linux shell:"<<std::endl;
    out << "    wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg.caffemodel"<<std::endl;
    out << "    wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_deploy.prototxt"<<std::endl;
    out << "    wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_labels.txt"<<std::endl<<std::endl;
    return out.str();
}

inline bool fileExists (std::string filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}
void textbox_draw(cv::Mat &src, std::vector<cv::Rect>  &groups,std::vector<float> &probs,std::vector<cv::String> wordList,float thres=0.6)
{
    for (int i=0;i<(int)groups.size(); i++)
    {
        if(probs[i]>thres)
        {
            if (src.type() == CV_8UC3)
            {
                cv::rectangle(src,groups.at(i).tl(),groups.at(i).br(),cv::Scalar( 0, 255, 255 ), 3, 8 );
                cv::putText(src, wordList[i],groups.at(i).tl() , cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar( 0,0,255 ));
            }
            else
                rectangle(src,groups.at(i).tl(),groups.at(i).br(),cv::Scalar( 255 ), 3, 8 );
        }
    }
}


int main(int argc, const char * argv[]){
    if(!cv::text::cnn_config::caffe_backend::getCaffeAvailable()){
        std::cout<<"The text module was compiled without Caffe which is the only available DeepCNN backend.\nAborting!\n";
        exit(1);
    }
    //set to true if you have a GPU with more than 3GB
    cv::text::cnn_config::caffe_backend::setCaffeGpuMode(true);

    if (argc < 3){
        std::cout<<getHelpStr(argv[0]);
        std::cout<<"Insufiecient parameters. Aborting!"<<std::endl;
        exit(1);
    }

    if (!fileExists("textbox.caffemodel") ||
            !fileExists("textbox_deploy.prototxt")){
           // !fileExists("dictnet_vgg_labels.txt"))

        std::cout<<getHelpStr(argv[0]);
        std::cout<<"Model files not found in the current directory. Aborting!"<<std::endl;
        exit(1);
    }

    if (fileExists(argv[1])){
        std::cout<<getHelpStr(argv[0]);
        std::cout<<"Output file must not exist. Aborting!"<<std::endl;
        exit(1);
    }

    cv::Mat image;
    image = cv::imread(cv::String(argv[2]));


    std::cout<<"Starting Text Box Demo"<<std::endl;
    cv::Ptr<cv::text::textDetector> textSpotter=cv::text::textDetector::create(
                "textbox_deploy.prototxt","textbox.caffemodel");

    //cv::Ptr<cv::text::textDetector> wordSpotter=
      //      cv::text::textDetector::create(cnn);
    std::cout<<"Created Text Spotter with text Boxes";

    std::vector<cv::Rect> bbox;
    std::vector<float> outProbabillities;
    textSpotter->textDetectInImage(image,bbox,outProbabillities);
   // textbox_draw(image, bbox,outProbabillities);
    float thres =0.6f;
    std::vector<cv::Mat> imageList;
    for(int imageIdx=0;imageIdx<(int)bbox.size();imageIdx++){
        if(outProbabillities[imageIdx]>thres){
            imageList.push_back(image(bbox.at(imageIdx)));
        }

    }
    // call dict net here for all detected parts
    cv::Ptr<cv::text::DeepCNN> cnn=cv::text::DeepCNN::createDictNet(
                "dictnet_vgg_deploy.prototxt","dictnet_vgg.caffemodel");

    cv::Ptr<cv::text::OCRHolisticWordRecognizer> wordSpotter=
            cv::text::OCRHolisticWordRecognizer::create(cnn,"dictnet_vgg_labels.txt");

    std::vector<cv::String> wordList;
    std::vector<double> wordProbabillities;
    wordSpotter->recogniseImageBatch(imageList,wordList,wordProbabillities);
    // write the output in file
    std::ofstream out;
    out.open(argv[1]);


    for (int i=0;i<(int)wordList.size(); i++)
    {
        cv::Point tl_ = bbox.at(i).tl();
        cv::Point br_ = bbox.at(i).br();

        out<<argv[2]<<","<<tl_.x<<","<<tl_.y<<","<<tl_.y<<","<<tl_.y<<","<<br_.x<<","<<br_.y<<","<<wordList[i]<<std::endl;

    }
    out.close();
    textbox_draw(image, bbox,outProbabillities,wordList);


    cv::imshow("TextBox Demo",image);
    std::cout << "Done!" << std::endl << std::endl;
    std::cout << "Press any key to exit." << std::endl << std::endl;
    if ((cv::waitKey()&0xff) == ' ')
        return 0;
}
