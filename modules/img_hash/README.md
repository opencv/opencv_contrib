## Synopsis

This module intend to port the algorithms from PHash library and implement other image hash algorithm do not exist in PHash library yet.

##Support Algorithms
- Average hash
    - somebody call it Different hash
- PHash
    - somebody call it Perceptual hash
- Marr Hildreth Hash
- Radial Variance Hash
- Block Mean Hash
    - support mode 0 and mode 1
- Color Moment Hash
    - This is the one and only hash algorithm resist to rotation attack(-90~90 degree) in img_hash

You can study more about image hashing from following paper and websites

* [Implementation and Benchmarking of Perceptual Image Hash Functions, im Juli 2010](http://www.phash.org/docs/pubs/thesis_zauner.pdf)
* [Looks Like It](http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html)

## Code Example

```cpp
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/img_hash.hpp>
#include <opencv2/imgproc.hpp>

void computeHash(cv::Ptr<cv::img_hash::ImgHashBase> algo)
{
    cv::Mat const input = cv::imread("lena.png");
    cv::Mat const target = cv::imread("lena_blur.png");

    cv::Mat inHash; //hash of input image
    cv::Mat targetHash; //hash of target image

    //comupte hash of input and target
    algo->compute(input, inHash);
    algo->compute(target, targetHash);
    //Compare the similarity of inHash and targetHash
    //recommended thresholds are written in the header files
    //of every classes
    double const mismatch = algo->compare(inHash, targetHash);
    std::cout<<mismatch<<std::endl;
}

int main()
{
    //disable opencl acceleration may(or may not) boost up speed of img_hash
    cv::ocl::setUseOpenCL(false);

    computeHash(img_hash::AverageHash::create());
    computeHash(img_hash::PHash::create());
    computeHash(img_hash::MarrHildrethHash::create());
    computeHash(img_hash::RadialVarianceHash::create());
    //BlockMeanHash support mode 0 and mode 1, they associate to
    //mode 1 and mode 2 of PHash library
    computeHash(img_hash::BlockMeanHash::create(0));
    computeHash(img_hash::BlockMeanHash::create(1));
    computeHash(img_hash::ColorMomentHash::create());
}
```

#Performance under different attacks

![Performance chart](https://3.bp.blogspot.com/-Li-zoGXC6-I/V3Wnp5tbFwI/AAAAAAAAA1Y/iVQkZmI6wWQcpxynuzW4FngJYVdXw3AtgCLcB/s1600/overall_result.JPG)

#Speed comparison with PHash library(100 images from ukbench)

![Hash Computation chart](https://3.bp.blogspot.com/-XIs-olyuK9Q/V3NKRDRzUiI/AAAAAAAAAwU/k99xuDGlCBYwO3ZDZNHcLweuaAt_cpHtwCLcB/s1600/Capture.JPG)
![Hash comparison chart](https://1.bp.blogspot.com/-anqfh2Awky4/V3NOOKvrQKI/AAAAAAAAAwo/pZjGDDnAPKooOZCCVnzGO4lJjKo7-KjlACLcB/s1600/Capture.JPG)

As you can see, hash computation speed of img_hash module outperform [PHash library](http://www.phash.org/) a lot.

PS : I do not list out the comparison of Average hash, PHash and Color Moment hash, because I cannot find them in PHash.

## Motivation

Collects useful image hash algorithms into opencv, so we do not need to rewrite them by ourselves again and again or rely on another 3rd party library(ex : PHash library). BOVW or correlation matching are good and robust, but they are very slow compare with image hash, if you need to deal with large scale CBIR(content based image retrieval) problem, image hash is a more reasonable solution.

## Installation

This module only depend on opencv_core and opencv_imgproc, by now it is not merged into the opencv_contrib yet, to build it, you can try following solution(please tell me a better one if you know, thanks)

1. git clone https://github.com/stereomatchingkiss/opencv_contrib
2. cd opencv_contrib
3. git branch -a //see all of the branch because part of them are hidden
4. git checkout -b img_hash origin/img_hash

### Solution A--copy to existing opencv folder

5. copy the whole folder opencv_contrib/modules/img_hash
6. check out the branch you want to work at
7. paste the folder img_hash you copied in to opencv_contrib/modules
8. build it as other opencv_contrib modules or you can add all of the headers and 
sources into your project

### Solution B--sync with master brnach of opencv_contrib
5. git pull https://github.com/opencv/opencv_contrib
6. build it as other opencv_contrib modules or you can add all of the headers and 
sources into your project

To study more details about those git commands, please go to [stack overflow](http://stackoverflow.com/questions/67699/clone-all-remote-branches-with-git)

## More info

You can learn more about img_hash modules from following links, these links show you how to find similar image from ukbench dataset, provide thorough benchmark of different attacks(contrast, blur, noise(gaussion,pepper and salt), jpeg compression, watermark, resize).

* [Introduction to image hash module of opencv](http://qtandopencv.blogspot.my/2016/06/introduction-to-image-hash-module-of.html)
* [Speed up image hashing of opencv(img_hash) and introduce color moment hash](http://qtandopencv.blogspot.my/2016/06/speed-up-image-hashing-of-opencvimghash.html)


## Contributors

Tham Ngap Wei, thamngapwei@gmail.com
