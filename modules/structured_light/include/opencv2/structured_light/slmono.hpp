// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_structured_light_HPP
#define OPENCV_structured_light_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cv{
namespace structured_light{
class CV_EXPORTS StructuredLightMono : public virtual Algorithm 
{
public:

    StructuredLightMono(Size img_size, int patterns, int stripes_number, std::string algs_type)
    {
        projector_size = img_size;
        pattern_num = patterns;
        alg_type = algs_type;
        stripes_num = stripes_number;
    }

    //generate patterns for projecting
    void generatePatterns(OutputArrayOfArrays patterns, float stripes_angle);

    //project patterns and capture with camera
    //CV_WRAP
//    void captureImages(InputArrayOfArrays patterns, OutputArrayOfArrays refs, OutputArrayOfArrays imgs, bool isCaptureRefs = true);

    //main phase unwrapping algorithm
    void unwrapPhase(InputOutputArrayOfArrays refs, InputOutputArrayOfArrays imgs, OutputArray out);

    //read references and phases from file
    //CV_WRAP
//    void readImages(std::vector<std::string> refs_files, std::vector<std::string> imgs_files, OutputArrayOfArrays refs, OutputArrayOfArrays imgs);

private:
    
    //size of the image for whole algorithm
    Size projector_size;

    //number of pattern used in SL algorithm starting from 3
    int pattern_num; 
    
    //number of stripes in the image pattern
    int stripes_num;

    //PCG or TPU 
    std::string alg_type;

    //remove shadows from images
    void removeShadows(InputOutputArrayOfArrays refs, InputOutputArrayOfArrays imgs); 

    //phase unwrapping with PCG algorithm based on DCT
    void computePhasePCG(InputOutputArrayOfArrays refs, InputOutputArrayOfArrays imgs, OutputArray out);

    //standart temporal unwrap algorithm 
    void computePhaseTPU(InputOutputArrayOfArrays refs, InputOutputArrayOfArrays imgs, OutputArray out);
};

}} // cv::structured_light::

#endif
