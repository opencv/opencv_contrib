/* Sample - Filtering
 * Target is to apply filtering using F-transform
 * on the image "input.png". Two different kernels
 * are used, where bigger radius (100 in this case)
 * means higher level of blurriness.
 *
 * Image "output1_filter.png" is created from "input.png"
 * using "kernel1" with radius 3.
 *
 * Image "output2_filter.png" is created from "input.png"
 * using "kernel2" with radius 100.
 *
 * Both kernels are created from linear function, using
 * linear interpolation (parametr ft:LINEAR).
 */

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/img_hash.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace cv;

void test_by_function(cv::Mat const &input, cv::Mat const &target);
void test_by_class(cv::Mat const &input, cv::Mat const &target);

int main(int argc, char **argv)
{
    if(argc != 3){
        std::cerr<<"must input the path of input "
                   "image and target image. ex : \n"
                   "hash_samples lena.jpg lena2.jpg\n";
    }

    Mat input = imread(argv[1]);
    Mat target = imread(argv[2]);

    test_by_function(input, target);
    test_by_class(input, target);

    return 0;
}

void test_by_function(cv::Mat const &input, cv::Mat const &target)
{
    cv::Mat inHash;
    cv::Mat outHash;

    img_hash::averageHash(input, inHash);
    img_hash::averageHash(target, outHash);
    //the lower the mismatch value, the better
    double const averageMismatch = norm(inHash, outHash, NORM_HAMMING);
    std::cout<<"averageMismatch : "<<averageMismatch<<std::endl;

    img_hash::pHash(input, inHash);
    img_hash::pHash(target, outHash);
    double const pHashMismatch = norm(inHash, outHash, NORM_HAMMING);
    std::cout<<"pHashMismatch : "<<pHashMismatch<<std::endl;

    img_hash::marrHildrethHash(input, inHash);
    img_hash::marrHildrethHash(target, outHash);
    double const marrMismatch = norm(inHash, outHash, NORM_HAMMING);
    std::cout<<"marrMismatch : "<<marrMismatch<<std::endl;

    //Please use the class version to compare the similarity
    //of the hash values of radialVarianceHash
    img_hash::radialVarianceHash(input, inHash);
    img_hash::radialVarianceHash(target, outHash);

    img_hash::blockMeanHash(input, inHash);
    img_hash::blockMeanHash(target, outHash);
    double const blockMisMatch = norm(inHash, outHash, NORM_HAMMING);
    std::cout<<"blockMisMatch : "<<blockMisMatch<<std::endl;
}

//benefits of using class is potential performance gain, because
//class will reuse the buffer, function will allocate new memory
//every time you call it
void test_by_class(cv::Mat const &input, cv::Mat const &target)
{
    cv::Mat inHash;
    cv::Mat outHash;

    Ptr<img_hash::ImgHashBase> hashFunc = img_hash::AverageHash::create();
    hashFunc->compute(input, inHash);
    hashFunc->compute(target, outHash);
    double const averageMismatch = hashFunc->compare(inHash, outHash);
    std::cout<<"averageMismatch : "<<averageMismatch<<std::endl;

    hashFunc = img_hash::PHash::create();
    hashFunc->compute(input, inHash);
    hashFunc->compute(target, outHash);
    double const pHashMismatch = hashFunc->compare(inHash, outHash);
    std::cout<<"pHashMismatch : "<<pHashMismatch<<std::endl;

    hashFunc = img_hash::MarrHildrethHash::create();
    hashFunc->compute(input, inHash);
    hashFunc->compute(target, outHash);
    double const marrMismatch = hashFunc->compare(inHash, outHash);
    std::cout<<"marrMismatch : "<<marrMismatch<<std::endl;

    hashFunc = img_hash::RadialVarianceHash::create();
    hashFunc->compute(input, inHash);
    hashFunc->compute(target, outHash);
    double const radialMismatch = hashFunc->compare(inHash, outHash);
    std::cout<<"radialMismatch : "<<radialMismatch<<std::endl;

    hashFunc = img_hash::BlockMeanHash::create();
    hashFunc->compute(input, inHash);
    hashFunc->compute(target, outHash);
    double const blockMeanMismatch = hashFunc->compare(inHash, outHash);
    std::cout<<"blockMeanMismatch : "<<blockMeanMismatch<<std::endl;
}
