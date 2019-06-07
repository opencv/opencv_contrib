// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_APRILTAG_HPP__
#define __OPENCV_APRILTAG_HPP__

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

namespace cv {
namespace apriltag {

enum AprilTagFamily {
    TAG_16h5,          ///< AprilTag 16h5 pattern
    TAG_25h9,          ///< AprilTag 25h9 pattern
    TAG_36h11,         ///< AprilTag 36h11 pattern
    TAG_CIRCLE21h7,    ///< AprilTag Circle21h7 pattern
    TAG_CIRCLE49h12,   ///< AprilTag Circle49h12 pattern
    TAG_CUSTOM48h12,   ///< AprilTag Custom48h12 pattern
    TAG_STANDARD41h12, ///< AprilTag Standard41h12 pattern
    TAG_STANDARD52h13  ///< AprilTag Standard52h13 pattern
};

class CV_EXPORTS AprilTagDetector {
public:
    explicit AprilTagDetector(const AprilTagFamily& tagFamily);
    ~AprilTagDetector();

    void detectTags(InputArray image, OutputArrayOfArrays corners, OutputArray ids);

    void drawDetectedTags(InputOutputArray image, InputArrayOfArrays corners,
                          InputArray ids = noArray(),
                          const Scalar& borderColor = Scalar(0, 255, 0),
                          int thickness = 2, double fontSize = 0.6);

    void estimateTagsPoseAprilTag(double tagSize, InputArray cameraMatrix, InputArray distCoeffs,
                                  std::vector<std::pair<Matx31d, Matx31d> >& rvecs,
                                  std::vector<std::pair<Matx31d, Matx31d> >& tvecs,
                                  std::vector<std::pair<double, double> >* reprojectionError = NULL,
                                  OutputArray objPoints = noArray());

    void estimateTagsPosePnP(InputArrayOfArrays corners, double tagSize,
                             InputArray cameraMatrix, InputArray distCoeffs,
                             std::vector<std::pair<Matx31d, Matx31d> >& rvecs,
                             std::vector<std::pair<Matx31d, Matx31d> >& tvecs,
                             SolvePnPMethod pnpMethod = SOLVEPNP_IPPE_SQUARE,
                             const std::vector<bool>& useExtrinsicGuesses = std::vector<bool>(),
                             std::vector<std::pair<double, double> >* reprojectionError = NULL,
                             OutputArray objectPoints = noArray());

    void estimateTagsPosePnP(InputArrayOfArrays corners, const std::vector<double>& tagsSize,
                             InputArray cameraMatrix, InputArray distCoeffs,
                             std::vector<std::pair<Matx31d, Matx31d> >& rvecs,
                             std::vector<std::pair<Matx31d, Matx31d> >& tvecs,
                             const std::vector<SolvePnPMethod>& pnpMethods,
                             const std::vector<bool>& useExtrinsicGuesses = std::vector<bool>(),
                             std::vector<std::pair<double, double> >* reprojectionError = NULL,
                             OutputArrayOfArrays objectsPoints = noArray());

    void drawTag(InputOutputArray image, const Size& size, int id);

    void setDecodeSharpening(double decodeSharpening);
    void setNumThreads(int nThreads);
    void setQuadDecimate(float quadDecimate);
    void setQuadSigma(float quadSigma);
    void setRefineEdges(bool refineEdges);

private:
    AprilTagDetector(const AprilTagDetector&);              // disabled
    AprilTagDetector& operator=(const AprilTagDetector&);   // disabled

    struct Impl;
    Impl* pImpl;
};

}
}

#endif

/* End of file. */
