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

std::string getHelpStr(std::string progFname){
    std::stringstream out;
    out << "    Demo of wordspotting CNN for text recognition." << std::endl;
    out << "    Max Jaderberg et al.: Reading Text in the Wild with Convolutional Neural Networks, IJCV 2015"<<std::endl<<std::endl;

    out << "    Usage: " << progFname << " <output_file> <input_image1> <input_image2> ... <input_imageN>" << std::endl;
    out << "    Caffe Model files  (dictnet_vgg.caffemodel, dictnet_vgg_deploy.prototxt, dictnet_vgg_labels.txt)"<<std::endl;
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


int main(int argc, const char * argv[]){
    const int USE_GPU=0;

    if (argc < 3){
        std::cout<<getHelpStr(argv[0]);
        exit(1);
        std::cout<<"Insufiecient parameters. Aborting!"<<std::endl;
    }

    if (!fileExists("dictnet_vgg.caffemodel") ||
            !fileExists("dictnet_vgg_deploy.prototxt") ||
            !fileExists("dictnet_vgg_labels.txt")){
        std::cout<<getHelpStr(argv[0]);
        std::cout<<"Model files not found in the current directory. Aborting!"<<std::endl;
        exit(1);
    }

    if (fileExists(argv[1])){
        std::cout<<getHelpStr(argv[0]);
        std::cout<<"Output file must not exist. Aborting!"<<std::endl;
        exit(1);
    }

    std::vector<cv::Mat> imageList;
    for(int imageIdx=2;imageIdx<argc;imageIdx++){
        if (fileExists(argv[imageIdx])){
            imageList.push_back(cv::imread(cv::String(argv[imageIdx])));
        }else{
            std::cout<<getHelpStr(argv[0]);
            std::cout<<argv[imageIdx]<<" doesn't exist. Aborting";
        }
    }
    cv::Ptr<cv::text::DictNet> cnn=cv::text::DictNet::create(
                "dictnet_vgg_deploy.prototxt","dictnet_vgg.caffemodel",100,USE_GPU);

    cv::Ptr<cv::text::OCRHolisticWordRecognizer> wordSpotter=
            cv::text::OCRHolisticWordRecognizer::create(cnn,"dictnet_vgg_labels.txt");

    std::vector<cv::String> wordList;
    std::vector<double> outProbabillities;
    wordSpotter->recogniseImageBatch(imageList,wordList,outProbabillities);

    std::ofstream out;
    out.open(argv[1]);
    for(int imgIdx=0;imgIdx<imageList.size();imgIdx++){
        out<<argv[imgIdx+2]<<","<<wordList[imgIdx]<<","<<outProbabillities[imgIdx]<<std::endl;
    }
    out.close();
}
