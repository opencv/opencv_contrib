// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_structured_light_mono_utils_HPP
#define OPENCV_structured_light_mono_utils_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

namespace cv{
namespace structured_light{

    //compute atan2 for object and reference images
    void computeAtanDiff(InputOutputArrayOfArrays src, OutputArray dst);

    /**
    Phase unwrapping algorithm based on PCG. 
    **/
    void unwrapPCG(InputArray img, OutputArray out, Size imgSize);

    /**
    Phase unwrapping algorithm based on TPU. 
    **/
    void unwrapTPU(InputArray phase1, InputArray phase2, OutputArray out, int scale);

    void lowPassFilter(InputArray img, OutputArray out, int filterSize = 30);
    void highPassFilter(InputArray img, OutputArray out, int filterSize = 30);

    void calibrateCameraProjector();
    void distortPatterns(); 
    void undistortPatterns();

    void savePointCloud(InputArray phase, string filename); //filter image from outliers and save as txt

    void circshift(OutputArray out, InputArray in, int xdim, int ydim, bool isFftshift);
    void createGrid(OutputArray output, Size size);
    void wrapSin(InputArray img, OutputArray out);
    void wrapCos(InputArray img, OutputArray out);
    void Laplacian(InputArray img, InputArray grid, OutputArray out, int flag);
    void computeDelta(InputArray img, InputArray grid, OutputArray out);
    void fft2(InputArray in, OutputArray complexI);
}
}

#endif