#ifndef OPENCV_ARUCO_FRACTAL_HPP
#define OPENCV_ARUCO_FRACTAL_HPP

#include <opencv2/core.hpp>
#include <opencv2/aruco/dictionary.hpp>

namespace cv {
namespace aruco {

class CV_EXPORTS_W FractalDictionary {
public:
    virtual ~FractalDictionary() = default;
};

CV_EXPORTS_W Ptr<FractalDictionary> getFractalDictionary(int dictType);

CV_EXPORTS_W void detectFractalMarkers(
    InputArray image,
    const Ptr<FractalDictionary>& fractalDict,
    OutputArrayOfArrays corners,
    OutputArray ids,
    const Ptr<DetectorParameters>& params = makePtr<DetectorParameters>(),
    OutputArrayOfArrays rejected = noArray());

} // namespace aruco
} // namespace cv

#endif
