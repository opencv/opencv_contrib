// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Author: Iago Suarez <iago.suarez.canosa@alumnos.upm.es>

// Implementation of the article:
//     Iago Suarez, Ghesn Sfeir, Jose M. Buenaposada, and Luis Baumela.
//     BEBLID: Boosted Efficient Binary Local Image Descriptor.
//     Pattern Recognition Letters, 133:366–372, 2020.

#include <bitset>
#include "precomp.hpp"

#define CV_BEBLID_PARALLEL

#define CV_ROUNDNUM(x) ((int)(x + 0.5f))
#define CV_DEGREES_TO_RADS 0.017453292519943295 // (M_PI / 180.0)
#define CV_BEBLID_EXTRA_RATIO_MARGIN 1.75f

using namespace cv;
using namespace std;


namespace cv
{
namespace xfeatures2d
{

// Struct containing the 6 parameters that define an Average Box weak-learner
struct ABWLParams
{
    int x1, y1, x2, y2, boxRadius, th;
};

// BEBLID implementation
class BEBLID_Impl CV_FINAL: public BEBLID
{
public:

    // constructor
    explicit BEBLID_Impl(float scale_factor, int n_bits = SIZE_512_BITS);

    // destructor
    ~BEBLID_Impl() CV_OVERRIDE = default;

    // returns the descriptor length in bytes
    int descriptorSize() const CV_OVERRIDE { return int(wl_params_.size() / 8); }

    // returns the descriptor type
    int descriptorType() const CV_OVERRIDE { return CV_8UC1; }

    // returns the default norm type
    int defaultNorm() const CV_OVERRIDE { return cv::NORM_HAMMING;  }

    // compute descriptors given keypoints
    void compute(InputArray image, vector<KeyPoint> &keypoints, OutputArray descriptors) CV_OVERRIDE;

private:
    std::vector<ABWLParams> wl_params_;
    float scale_factor_;
    cv::Size patch_size_;

    void computeBEBLID(const cv::Mat &integralImg,
                       const std::vector<cv::KeyPoint> &keypoints,
                       cv::Mat &descriptors);
}; // END BEBLID_Impl CLASS

/**
 * @brief Function that determines if a keypoint is close to the image border.
 * @param kp The detected keypoint
 * @param imgSize The size of the image
 * @param patchSize The size of the normalized patch where the measurement functions were learnt.
 * @param scaleFactor A scale factor that magnifies the measurement functions w.r.t. the keypoint.
 * @return true if the keypoint is in the border, false otherwise
 */
inline bool isKeypointInTheBorder(const cv::KeyPoint &kp,
                                  const cv::Size &imgSize,
                                  const cv::Size &patchSize = {32, 32},
                                  float scaleFactor = 1)
{
    // This would be the correct measure but since we will compare with half of the size, use this as border size
    float s = scaleFactor * kp.size / (patchSize.width + patchSize.height);
    cv::Size2f border(patchSize.width * s * CV_BEBLID_EXTRA_RATIO_MARGIN,
                      patchSize.height * s * CV_BEBLID_EXTRA_RATIO_MARGIN);

    if (kp.pt.x < border.width || kp.pt.x + border.width >= imgSize.width) return true;
    if (kp.pt.y < border.height || kp.pt.y + border.height >= imgSize.height) return true;
    return false;
}

/**
 * @brief Rectifies the coordinates of the measurement functions that conform the descriptor
 * with the keypoint location parameters.
 * @param wlPatchParams The input weak learner parameters learnt for the normalized patch
 * @param wlImageParams The output  weak learner parameters adapted to the keypoint location
 * @param kp The keypoint defining the offset, rotation and scale to be applied
 * @param scaleFactor A scale factor that magnifies the measurement functions w.r.t. the keypoint.
 * @param patchSize The size of the normalized patch where the measurement functions were learnt.
 */
inline void rectifyABWL(const std::vector<ABWLParams> &wlPatchParams,
                        std::vector<ABWLParams> &wlImageParams,
                        const cv::KeyPoint &kp,
                        float scaleFactor = 1,
                        const cv::Size &patchSize = cv::Size(32, 32))
{
    float m00, m01, m02, m10, m11, m12;
    float s, cosine, sine;
    uint32_t i;

    s = scaleFactor * kp.size / (0.5f * (patchSize.width + patchSize.height));
    wlImageParams.resize(wlPatchParams.size());

    if (kp.angle == -1)
    {
        m00 = s;
        m01 = 0.0f;
        m02 = -0.5f * s * patchSize.width + kp.pt.x;
        m10 = 0.0f;
        m11 = s;
        m12 = -s * 0.5f * patchSize.height + kp.pt.y;
    }
    else
    {
        cosine = (kp.angle >= 0) ? float(cos(kp.angle * CV_DEGREES_TO_RADS)) : 1.f;
        sine = (kp.angle >= 0) ? float(sin(kp.angle * CV_DEGREES_TO_RADS)) : 0.f;

        m00 = s * cosine;
        m01 = -s * sine;
        m02 = (-s * cosine + s * sine) * patchSize.width * 0.5f + kp.pt.x;
        m10 = s * sine;
        m11 = s * cosine;
        m12 = (-s * sine - s * cosine) * patchSize.height * 0.5f + kp.pt.y;
    }

    for (i = 0; i < wlPatchParams.size(); i++)
    {
        wlImageParams[i].x1 = CV_ROUNDNUM(m00 * wlPatchParams[i].x1 + m01 * wlPatchParams[i].y1 + m02);
        wlImageParams[i].y1 = CV_ROUNDNUM(m10 * wlPatchParams[i].x1 + m11 * wlPatchParams[i].y1 + m12);
        wlImageParams[i].x2 = CV_ROUNDNUM(m00 * wlPatchParams[i].x2 + m01 * wlPatchParams[i].y2 + m02);
        wlImageParams[i].y2 = CV_ROUNDNUM(m10 * wlPatchParams[i].x2 + m11 * wlPatchParams[i].y2 + m12);
        wlImageParams[i].boxRadius = CV_ROUNDNUM(s * wlPatchParams[i].boxRadius);
    }
}

/**
 * @brief Computes the Average Box Weak-Learner response, measuring the difference of
 * gray level in the two square regions.
 * @param wlImageParams The weak-learner parameter defining the size and locations of each box.
 * @param integralImage The integral image used to compute the average gray value in the square regions.
 * @return The difference of gray level in the two squares defined by wlImageParams
 */
inline float computeABWLResponse(const ABWLParams &wlImageParams, const cv::Mat &integralImage)
{
    CV_DbgAssert(!integralImage.empty());

    int frameWidth, frameHeight, box1x1, box1y1, box1x2, box1y2, box2x1, box2y1, box2x2, box2y2;
    int idx1, idx2, idx3, idx4;
    int A, B, C, D;
    const int *ptr;
    int box_area1, box_area2, width;
    float sum1, sum2, average1, average2;
    // Since the integral image has one extra row and col, calculate the patch dimensions
    frameWidth = integralImage.cols;
    frameHeight = integralImage.rows;

    // For the first box, we calculate its margin coordinates
    box1x1 = wlImageParams.x1 - wlImageParams.boxRadius;
    if (box1x1 < 0) box1x1 = 0;
    else if (box1x1 >= frameWidth - 1) box1x1 = frameWidth - 2;
    box1y1 = wlImageParams.y1 - wlImageParams.boxRadius;
    if (box1y1 < 0) box1y1 = 0;
    else if (box1y1 >= frameHeight - 1) box1y1 = frameHeight - 2;
    box1x2 = wlImageParams.x1 + wlImageParams.boxRadius + 1;
    if (box1x2 <= 0) box1x2 = 1;
    else if (box1x2 >= frameWidth) box1x2 = frameWidth - 1;
    box1y2 = wlImageParams.y1 + wlImageParams.boxRadius + 1;
    if (box1y2 <= 0) box1y2 = 1;
    else if (box1y2 >= frameHeight) box1y2 = frameHeight - 1;
    CV_DbgAssert((box1x1 < box1x2 && box1y1 < box1y2) && "Box 1 has size 0");

    // For the second box, we calculate its margin coordinates
    box2x1 = wlImageParams.x2 - wlImageParams.boxRadius;
    if (box2x1 < 0) box2x1 = 0;
    else if (box2x1 >= frameWidth - 1) box2x1 = frameWidth - 2;
    box2y1 = wlImageParams.y2 - wlImageParams.boxRadius;
    if (box2y1 < 0) box2y1 = 0;
    else if (box2y1 >= frameHeight - 1) box2y1 = frameHeight - 2;
    box2x2 = wlImageParams.x2 + wlImageParams.boxRadius + 1;
    if (box2x2 <= 0) box2x2 = 1;
    else if (box2x2 >= frameWidth) box2x2 = frameWidth - 1;
    box2y2 = wlImageParams.y2 + wlImageParams.boxRadius + 1;
    if (box2y2 <= 0) box2y2 = 1;
    else if (box2y2 >= frameHeight) box2y2 = frameHeight - 1;
    CV_DbgAssert((box2x1 < box2x2 && box2y1 < box2y2) && "Box 2 has size 0");

    // Calculate the indices on the integral image where the box falls
    width = integralImage.cols;
    idx1 = box1y1 * width + box1x1;
    idx2 = box1y1 * width + box1x2;
    idx3 = box1y2 * width + box1x1;
    idx4 = box1y2 * width + box1x2;
    CV_DbgAssert(idx1 >= 0 && idx1 < integralImage.size().area());
    CV_DbgAssert(idx2 >= 0 && idx2 < integralImage.size().area());
    CV_DbgAssert(idx3 >= 0 && idx3 < integralImage.size().area());
    CV_DbgAssert(idx4 >= 0 && idx4 < integralImage.size().area());
    ptr = integralImage.ptr<int>();

    // Read the integral image values for the first box
    A = ptr[idx1];
    B = ptr[idx2];
    C = ptr[idx3];
    D = ptr[idx4];

    // Calculate the mean intensity value of the pixels in the box
    sum1 = float(A + D - B - C);
    box_area1 = (box1y2 - box1y1) * (box1x2 - box1x1);
    CV_DbgAssert(box_area1 > 0);
    average1 = sum1 / box_area1;

    // Calculate the indices on the integral image where the box falls
    idx1 = box2y1 * width + box2x1;
    idx2 = box2y1 * width + box2x2;
    idx3 = box2y2 * width + box2x1;
    idx4 = box2y2 * width + box2x2;

    CV_DbgAssert(idx1 >= 0 && idx1 < integralImage.size().area());
    CV_DbgAssert(idx2 >= 0 && idx2 < integralImage.size().area());
    CV_DbgAssert(idx3 >= 0 && idx3 < integralImage.size().area());
    CV_DbgAssert(idx4 >= 0 && idx4 < integralImage.size().area());

    // Read the integral image values for the first box
    A = ptr[idx1];
    B = ptr[idx2];
    C = ptr[idx3];
    D = ptr[idx4];

    // Calculate the mean intensity value of the pixels in the box
    sum2 = float(A + D - B - C);
    box_area2 = (box2y2 - box2y1) * (box2x2 - box2x1);
    CV_DbgAssert(box_area2 > 0);
    average2 = sum2 / box_area2;

    return average1 - average2;
}

// descriptor computation using keypoints
void BEBLID_Impl::compute(InputArray _image, vector<KeyPoint> &keypoints, OutputArray _descriptors)
{
    Mat image = _image.getMat();

    if (image.empty())
        return;

    if (keypoints.empty())
        return;

    Mat grayImage;
    switch (image.type()) {
    case CV_8UC1:
        grayImage = image;
        break;
    case CV_8UC3:
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
        break;
    case CV_8UC4:
        cvtColor(image, grayImage, COLOR_BGRA2GRAY);
        break;
    default:
        CV_Error(Error::StsBadArg, "Image should be 8UC1, 8UC3 or 8UC4");
    }

    cv::Mat integralImg;

    // compute the integral image
    cv::integral(grayImage, integralImg);

    // Create the output array of descriptors
    _descriptors.create((int)keypoints.size(), descriptorSize(), descriptorType());

    // descriptor storage
    cv::Mat descriptors = _descriptors.getMat();
    CV_DbgAssert(descriptors.type() == CV_8UC1);

    // Compute the BEBLID descriptors
    computeBEBLID(integralImg, keypoints, descriptors);
}

// constructor
BEBLID_Impl::BEBLID_Impl(float scale_factor, int n_bits)
    : scale_factor_(scale_factor), patch_size_(32, 32)
{
    // Populate wl_params_
    if (n_bits == SIZE_512_BITS)
    {
        // Pre-trained parameters of BEBLID-512 trained in Liberty data set with
        // a million of patch pairs, 20% positives and 80% negatives
        wl_params_ =
            {{24, 18, 15, 17, 6, 13}, {19, 14, 13, 17, 2, 18}, {23, 19, 12, 15, 6, 19}, {24, 14, 16, 16, 6, 11},
             {16, 15, 12, 16, 1, 12}, {16, 15, 7, 10, 4, 10}, {17, 12, 8, 17, 3, 16}, {24, 12, 11, 17, 7, 19},
             {19, 17, 14, 11, 3, 13}, {16, 15, 13, 15, 1, 10}, {16, 14, 6, 18, 5, 10}, {25, 5, 14, 15, 5, 15},
             {17, 18, 14, 16, 2, 10}, {17, 14, 14, 13, 2, 9}, {15, 14, 6, 22, 5, 7}, {14, 16, 5, 17, 5, 5},
             {16, 13, 15, 16, 1, 4}, {18, 17, 15, 15, 1, 9}, {26, 26, 15, 14, 5, 12}, {18, 18, 16, 16, 1, 4},
             {15, 14, 14, 27, 4, 0}, {17, 13, 15, 16, 1, 6}, {15, 15, 13, 14, 1, 6}, {18, 17, 16, 16, 1, 4},
             {14, 13, 6, 7, 5, 4}, {27, 12, 17, 15, 4, 8}, {12, 13, 7, 24, 7, 2}, {17, 18, 15, 15, 1, 6},
             {16, 16, 12, 17, 1, 12}, {27, 20, 16, 16, 4, 11}, {12, 14, 7, 5, 5, 0}, {12, 16, 7, 26, 5, 0},
             {15, 15, 15, 7, 4, -1}, {16, 17, 14, 17, 2, 6}, {16, 13, 10, 6, 4, 7}, {15, 26, 15, 19, 4, 1},
             {26, 5, 17, 13, 5, 7}, {15, 23, 5, 12, 5, 8}, {17, 14, 10, 11, 3, 14}, {21, 27, 17, 16, 4, 5},
             {15, 16, 14, 16, 1, 3}, {14, 11, 12, 26, 5, 1}, {12, 14, 12, 5, 4, -3}, {16, 16, 14, 12, 1, 7},
             {13, 20, 7, 13, 3, 4}, {19, 6, 17, 16, 6, 3}, {11, 9, 10, 19, 4, 2}, {14, 15, 13, 9, 3, 1},
             {16, 16, 14, 25, 3, 3}, {8, 26, 8, 13, 4, 3}, {16, 14, 15, 19, 2, 3}, {18, 15, 15, 16, 1, 9},
             {26, 23, 19, 16, 5, 4}, {11, 21, 4, 13, 4, 1}, {20, 16, 20, 5, 4, 2}, {15, 16, 15, 13, 1, 0},
             {16, 20, 16, 15, 2, 0}, {22, 13, 17, 14, 2, 8}, {18, 17, 14, 15, 1, 13}, {21, 12, 20, 26, 4, 3},
             {10, 7, 8, 18, 5, 3}, {11, 26, 11, 20, 5, 2}, {13, 21, 13, 17, 3, 1}, {10, 23, 6, 7, 6, 1},
             {10, 14, 5, 14, 5, 0}, {23, 25, 16, 6, 6, 8}, {18, 16, 18, 5, 4, 1}, {16, 16, 16, 14, 1, 0},
             {11, 15, 4, 23, 4, -2}, {17, 14, 16, 16, 1, 2}, {26, 4, 20, 24, 4, 2}, {20, 19, 18, 14, 2, 3},
             {14, 17, 10, 15, 2, 6}, {17, 13, 17, 9, 3, 0}, {26, 21, 5, 24, 5, 20}, {20, 15, 19, 25, 5, 3},
             {27, 15, 19, 5, 4, 5}, {10, 14, 10, 6, 6, -2}, {12, 22, 11, 10, 3, 2}, {17, 16, 16, 20, 2, 3},
             {15, 15, 12, 19, 1, 7}, {15, 11, 14, 17, 2, 4}, {14, 20, 10, 15, 2, 7}, {10, 14, 3, 7, 3, -5},
             {12, 16, 9, 11, 3, 1}, {19, 17, 17, 11, 2, 5}, {26, 7, 19, 26, 5, 4}, {20, 10, 19, 18, 3, 1},
             {17, 13, 16, 16, 1, 2}, {17, 11, 16, 4, 4, 2}, {15, 19, 14, 12, 2, 3}, {17, 18, 16, 13, 1, 3},
             {11, 9, 4, 27, 4, -1}, {21, 23, 18, 17, 3, 3}, {7, 21, 6, 7, 5, -1}, {25, 27, 21, 18, 4, -1},
             {14, 17, 14, 14, 2, 0}, {12, 11, 8, 19, 3, 3}, {14, 15, 13, 22, 2, 0}, {8, 23, 5, 17, 5, 1},
             {15, 16, 14, 8, 2, 1}, {16, 24, 15, 18, 3, 3}, {19, 25, 19, 18, 5, -1}, {11, 23, 10, 13, 2, 3},
             {19, 14, 18, 22, 2, 3}, {26, 15, 22, 6, 4, 2}, {24, 17, 19, 8, 3, 5}, {21, 15, 16, 15, 1, 10},
             {15, 14, 14, 20, 1, 2}, {16, 27, 13, 5, 4, 5}, {10, 4, 5, 13, 4, 3}, {12, 14, 10, 10, 2, 0},
             {14, 18, 14, 11, 1, -1}, {23, 6, 22, 20, 5, 0}, {14, 12, 10, 19, 2, 6}, {17, 18, 17, 15, 2, 0},
             {16, 15, 15, 18, 1, 4}, {11, 13, 3, 4, 3, -4}, {15, 14, 15, 8, 2, -1}, {11, 23, 5, 26, 5, 0},
             {20, 20, 19, 17, 2, 1}, {22, 19, 19, 20, 2, 3}, {16, 5, 15, 24, 4, 2}, {18, 15, 16, 12, 1, 5},
             {28, 27, 23, 15, 3, -2}, {7, 25, 6, 18, 6, 2}, {12, 19, 12, 13, 3, 0}, {9, 7, 4, 17, 4, 1},
             {14, 18, 13, 12, 1, 2}, {13, 16, 10, 23, 2, 1}, {24, 25, 23, 13, 6, -1}, {8, 13, 7, 4, 4, -3},
             {17, 15, 17, 11, 2, 0}, {20, 13, 18, 15, 1, 3}, {28, 3, 23, 15, 3, -2}, {13, 17, 12, 11, 1, 0},
             {16, 18, 16, 11, 1, 0}, {26, 16, 24, 26, 5, 2}, {14, 14, 11, 15, 1, 6}, {15, 9, 15, 3, 3, -1},
             {12, 28, 10, 19, 3, 6}, {18, 17, 18, 14, 2, 0}, {16, 14, 14, 15, 1, 7}, {20, 18, 19, 10, 2, 2},
             {27, 28, 18, 24, 3, 4}, {15, 11, 14, 25, 2, 1}, {16, 18, 15, 16, 1, 3}, {5, 27, 4, 6, 4, 0},
             {17, 20, 17, 14, 1, 0}, {13, 15, 9, 14, 1, 3}, {9, 23, 3, 23, 3, -1}, {9, 10, 3, 9, 3, -2},
             {16, 27, 16, 9, 3, 0}, {13, 17, 11, 15, 1, 3}, {14, 18, 14, 15, 1, 0}, {28, 12, 20, 21, 3, 2},
             {23, 7, 4, 27, 4, 16}, {16, 18, 16, 16, 1, -1}, {13, 16, 12, 19, 1, 1}, {20, 11, 19, 18, 2, 1},
             {23, 14, 19, 13, 1, 2}, {23, 10, 19, 3, 3, 5}, {15, 18, 13, 15, 1, 6}, {8, 14, 3, 19, 3, -3},
             {7, 18, 3, 17, 3, -2}, {22, 4, 21, 7, 4, 0}, {3, 28, 3, 18, 3, 2}, {19, 20, 17, 14, 1, 4},
             {16, 22, 15, 6, 2, 2}, {22, 20, 19, 29, 2, 5}, {11, 21, 9, 14, 2, 2}, {7, 9, 6, 4, 4, -2},
             {26, 19, 23, 9, 4, 1}, {16, 17, 16, 12, 2, 0}, {15, 5, 3, 4, 3, 4}, {18, 14, 17, 17, 1, 2},
             {19, 11, 17, 13, 1, 4}, {11, 17, 10, 10, 2, -1}, {15, 23, 12, 29, 2, 3}, {28, 20, 24, 17, 3, -1},
             {13, 10, 11, 2, 2, -1}, {28, 11, 23, 15, 3, -1}, {16, 21, 16, 20, 2, 0}, {8, 8, 7, 17, 2, 2},
             {15, 19, 14, 16, 1, 4}, {17, 11, 17, 10, 2, 0}, {22, 21, 19, 16, 1, 1}, {13, 17, 13, 14, 1, 0},
             {19, 13, 18, 16, 1, 2}, {6, 25, 5, 27, 4, -1}, {16, 29, 16, 22, 2, 0}, {23, 27, 23, 22, 4, -1},
             {29, 2, 22, 10, 2, -1}, {22, 10, 22, 5, 5, 1}, {20, 16, 19, 15, 1, 1}, {20, 9, 19, 14, 1, 0},
             {29, 29, 23, 22, 2, -1}, {12, 11, 10, 18, 1, 3}, {4, 16, 4, 2, 2, -2}, {14, 8, 13, 2, 2, 0},
             {16, 3, 15, 6, 3, 2}, {23, 8, 15, 2, 2, 10}, {18, 19, 18, 16, 1, 0}, {12, 21, 6, 18, 1, 2},
             {18, 15, 16, 19, 1, 5}, {16, 21, 16, 8, 2, 0}, {18, 26, 17, 23, 2, 1}, {7, 8, 3, 3, 3, -3},
             {6, 24, 3, 28, 3, -2}, {10, 19, 9, 26, 2, -3}, {17, 9, 16, 13, 1, 2}, {13, 15, 13, 10, 1, -2},
             {18, 16, 18, 12, 1, 0}, {17, 13, 17, 11, 1, 0}, {6, 16, 3, 12, 3, -2}, {15, 21, 15, 20, 1, 0},
             {23, 17, 20, 15, 2, 1}, {28, 22, 25, 8, 3, 0}, {5, 16, 3, 25, 3, -3}, {14, 13, 13, 20, 1, 2},
             {28, 28, 20, 27, 3, 2}, {15, 29, 8, 25, 2, 7}, {10, 28, 5, 24, 3, 2}, {19, 14, 18, 13, 1, 2},
             {19, 26, 14, 28, 3, 7}, {18, 21, 17, 18, 1, 2}, {13, 17, 9, 20, 1, 2}, {15, 13, 13, 11, 1, 4},
             {27, 7, 25, 15, 4, -1}, {12, 15, 11, 17, 1, 1}, {13, 20, 12, 15, 1, 3}, {15, 20, 14, 22, 1, 2},
             {19, 29, 17, 27, 2, 2}, {19, 3, 18, 5, 3, 1}, {9, 21, 9, 17, 2, 1}, {19, 18, 17, 18, 1, 4},
             {25, 13, 24, 18, 3, 0}, {11, 15, 10, 13, 1, 0}, {9, 9, 8, 3, 2, -2}, {6, 8, 3, 8, 3, -1},
             {28, 19, 23, 28, 3, 2}, {10, 30, 9, 23, 1, 3}, {5, 5, 3, 18, 3, 1}, {14, 17, 12, 20, 1, 3},
             {29, 16, 23, 15, 2, -1}, {23, 15, 21, 22, 2, 2}, {28, 3, 25, 5, 3, 0}, {12, 20, 11, 17, 1, 2},
             {20, 22, 18, 20, 1, 2}, {5, 9, 2, 2, 2, -3}, {7, 27, 3, 19, 3, 1}, {13, 2, 7, 6, 2, 4},
             {18, 29, 17, 25, 2, 1}, {15, 21, 14, 17, 1, 4}, {13, 29, 12, 26, 2, 2}, {5, 22, 4, 12, 2, 0},
             {16, 21, 16, 11, 1, 0}, {16, 23, 16, 10, 1, 0}, {11, 5, 10, 11, 2, 3}, {15, 10, 14, 21, 1, 3},
             {10, 18, 9, 18, 1, 0}, {17, 9, 16, 5, 2, 2}, {19, 19, 19, 12, 1, 0}, {25, 12, 22, 4, 2, 2},
             {6, 18, 1, 20, 1, -3}, {10, 13, 10, 10, 2, -1}, {25, 16, 22, 16, 1, 0}, {18, 13, 18, 12, 1, 0},
             {14, 13, 12, 11, 1, 3}, {10, 27, 1, 29, 1, -1}, {13, 8, 11, 6, 1, 1}, {24, 24, 21, 28, 3, 2},
             {22, 17, 20, 17, 1, 1}, {12, 13, 11, 18, 1, 1}, {23, 3, 21, 7, 3, 0}, {18, 12, 17, 13, 1, 2},
             {7, 28, 7, 25, 3, 1}, {28, 28, 28, 15, 3, -1}, {17, 7, 17, 2, 2, 0}, {19, 9, 17, 11, 1, 3},
             {14, 23, 14, 9, 1, 0}, {7, 22, 7, 19, 2, 1}, {29, 24, 29, 2, 2, 0}, {28, 15, 25, 11, 3, 0},
             {5, 11, 1, 10, 1, -2}, {2, 22, 2, 2, 2, -1}, {22, 30, 16, 27, 1, 5}, {20, 15, 19, 13, 1, 1},
             {23, 19, 22, 14, 2, 0}, {5, 7, 5, 3, 3, -1}, {19, 20, 18, 18, 1, 1}, {29, 9, 25, 13, 2, -1},
             {29, 23, 26, 23, 2, 0}, {9, 13, 8, 8, 1, -2}, {21, 22, 21, 18, 2, -1}, {29, 12, 28, 20, 2, 0},
             {18, 5, 1, 4, 1, 9}, {17, 4, 17, 2, 2, 0}, {28, 29, 24, 25, 2, 0}, {14, 23, 13, 29, 1, 0},
             {13, 5, 13, 1, 1, -1}, {20, 25, 20, 21, 1, -1}, {6, 5, 2, 11, 2, 0}, {10, 14, 9, 21, 1, -1},
             {13, 16, 13, 14, 1, 0}, {19, 17, 18, 14, 1, 2}, {14, 21, 14, 17, 1, 1}, {20, 10, 18, 12, 1, 2},
             {20, 4, 19, 3, 3, 1}, {3, 15, 1, 30, 1, -3}, {13, 4, 8, 1, 1, 2}, {10, 18, 9, 14, 1, 0},
             {6, 15, 1, 12, 1, -3}, {10, 25, 10, 20, 1, 2}, {14, 11, 14, 7, 1, -1}, {22, 9, 20, 4, 1, 2},
             {15, 27, 8, 30, 1, 4}, {10, 5, 10, 2, 2, -1}, {17, 16, 16, 12, 1, 3}, {15, 18, 15, 10, 1, -1},
             {20, 30, 20, 23, 1, -1}, {14, 9, 13, 22, 1, 2}, {14, 22, 12, 25, 1, 2}, {5, 23, 2, 23, 2, -1},
             {10, 16, 9, 16, 1, 0}, {26, 2, 19, 4, 1, 2}, {3, 23, 2, 13, 2, 0}, {3, 17, 3, 7, 2, -1},
             {15, 26, 15, 23, 1, 0}, {22, 14, 22, 8, 1, 1}, {28, 9, 27, 6, 3, 0}, {26, 22, 25, 28, 3, 1},
             {17, 10, 17, 5, 1, 1}, {11, 21, 10, 17, 1, 2}, {20, 18, 20, 16, 1, 0}, {7, 20, 5, 20, 1, -1},
             {17, 24, 17, 8, 1, 0}, {24, 9, 20, 9, 1, 1}, {4, 13, 1, 16, 1, -1}, {30, 1, 28, 16, 1, -1},
             {17, 21, 17, 17, 1, 0}, {19, 4, 11, 2, 1, 9}, {30, 5, 24, 6, 1, 0}, {22, 19, 22, 12, 1, 0},
             {9, 16, 9, 12, 1, -1}, {12, 16, 12, 12, 1, -1}, {12, 24, 11, 29, 1, -1}, {3, 6, 1, 4, 1, -1},
             {23, 29, 20, 27, 2, 1}, {23, 17, 22, 16, 1, 0}, {30, 20, 26, 22, 1, 0}, {9, 2, 6, 5, 2, 1},
             {20, 17, 19, 16, 1, 1}, {18, 26, 17, 30, 1, 1}, {29, 14, 28, 14, 2, 0}, {20, 13, 19, 14, 1, 1},
             {15, 23, 15, 21, 1, 0}, {8, 26, 2, 30, 1, -2}, {4, 5, 3, 2, 2, -1}, {7, 16, 6, 12, 1, -1},
             {29, 9, 23, 2, 2, 1}, {13, 2, 12, 5, 2, 2}, {20, 18, 19, 21, 1, 2}, {7, 29, 2, 25, 2, 0},
             {20, 3, 18, 8, 1, 1}, {14, 14, 14, 11, 1, -1}, {12, 12, 12, 10, 1, -1}, {17, 27, 15, 30, 1, 2},
             {22, 27, 20, 29, 2, 1}, {7, 12, 5, 9, 1, -2}, {30, 30, 24, 24, 1, 0}, {19, 3, 19, 2, 2, 0},
             {13, 19, 12, 18, 1, 2}, {3, 30, 2, 24, 1, 1}, {9, 14, 7, 19, 1, -1}, {17, 22, 17, 18, 1, 0},
             {18, 24, 17, 22, 1, 1}, {2, 18, 1, 23, 1, -1}, {30, 23, 24, 19, 1, -1}, {11, 10, 11, 5, 1, -2},
             {9, 30, 9, 27, 1, 1}, {21, 13, 20, 8, 1, 2}, {6, 3, 2, 2, 2, -1}, {23, 22, 22, 26, 1, 1},
             {12, 26, 11, 25, 1, 1}, {22, 1, 19, 5, 1, 1}, {4, 24, 1, 25, 1, -1}, {5, 13, 5, 7, 1, -1},
             {26, 22, 24, 16, 1, -1}, {27, 8, 27, 3, 2, 0}, {13, 18, 13, 16, 1, 0}, {19, 15, 18, 17, 1, 2},
             {30, 29, 26, 28, 1, 0}, {20, 15, 20, 14, 1, 0}, {3, 18, 1, 15, 1, -1}, {18, 11, 17, 10, 1, 2},
             {4, 18, 4, 16, 1, 0}, {8, 27, 5, 30, 1, -1}, {30, 15, 28, 22, 1, 0}, {9, 19, 8, 22, 1, -1},
             {30, 4, 29, 4, 1, 0}, {17, 10, 17, 8, 1, 0}, {22, 6, 22, 1, 1, 1}, {2, 11, 1, 15, 1, 0},
             {3, 16, 1, 17, 1, -1}, {9, 3, 8, 2, 2, 0}, {3, 11, 1, 10, 1, -1}, {16, 29, 15, 28, 1, 1},
             {15, 20, 15, 19, 1, 0}, {20, 17, 19, 17, 1, 1}, {10, 3, 9, 8, 1, 2}, {10, 22, 7, 26, 1, -1},
             {8, 16, 6, 16, 1, -1}, {16, 28, 16, 25, 1, 0}, {12, 25, 10, 21, 1, 3}, {8, 9, 7, 7, 1, -1},
             {3, 1, 1, 6, 1, 0}, {16, 7, 15, 9, 1, 2}, {30, 23, 29, 23, 1, 0}, {22, 24, 21, 29, 1, 1},
             {15, 1, 14, 3, 1, 1}, {18, 6, 17, 9, 1, 1}, {26, 25, 25, 19, 1, -1}, {25, 13, 22, 18, 1, 0},
             {11, 1, 10, 3, 1, 1}, {29, 28, 28, 30, 1, 0}, {16, 17, 16, 13, 5, 0}, {28, 18, 28, 12, 2, 0},
             {3, 22, 1, 23, 1, -1}, {10, 11, 10, 9, 1, -1}, {7, 13, 6, 20, 1, -1}, {1, 15, 1, 6, 1, -1},
             {16, 12, 16, 11, 1, 0}, {3, 26, 2, 30, 1, -1}, {28, 30, 26, 23, 1, -1}, {17, 22, 16, 25, 1, 2},
             {30, 13, 26, 7, 1, 0}, {10, 8, 7, 10, 1, 1}, {2, 27, 1, 22, 1, 0}, {30, 7, 27, 8, 1, 0},
             {22, 19, 21, 22, 1, 1}, {5, 19, 4, 21, 1, -1}, {24, 6, 23, 11, 1, -1}, {24, 17, 23, 14, 1, 0},
             {30, 7, 28, 1, 1, 0}, {11, 16, 11, 15, 1, 0}, {29, 2, 26, 4, 1, 0}, {20, 4, 18, 1, 1, 2},
             {18, 2, 17, 3, 1, 1}, {20, 30, 18, 29, 1, 1}, {29, 15, 29, 9, 2, 0}, {14, 8, 14, 5, 1, -1},
             {17, 15, 16, 18, 1, 3}, {12, 4, 11, 2, 2, 0}, {23, 8, 21, 11, 1, 0}, {8, 30, 7, 24, 1, 2},
             {2, 20, 1, 16, 1, 0}, {15, 26, 14, 29, 1, 1}, {4, 30, 3, 29, 1, 0}, {19, 17, 19, 16, 1, 0},
             {13, 17, 13, 15, 1, 0}, {2, 9, 1, 1, 1, -1}, {30, 28, 27, 27, 1, 0}, {27, 4, 26, 1, 1, 0},
             {19, 23, 19, 20, 1, -1}, {15, 24, 15, 23, 1, 0}, {2, 29, 1, 28, 1, 0}, {2, 5, 1, 6, 1, 0},
             {24, 29, 23, 26, 1, 0}, {13, 12, 12, 11, 1, 1}, {12, 17, 12, 15, 1, 0}, {24, 26, 24, 22, 1, -1},
             {11, 3, 10, 5, 1, 1}, {30, 2, 30, 1, 1, 0}, {18, 30, 18, 29, 1, 0}, {30, 25, 29, 29, 1, 0},
             {12, 30, 10, 28, 1, 1}, {24, 12, 22, 14, 1, 0}, {6, 13, 4, 15, 1, -1}, {2, 26, 2, 23, 1, 0},
             {8, 9, 7, 13, 1, 1}, {30, 1, 27, 1, 1, 0}, {26, 29, 24, 30, 1, 0}, {18, 11, 18, 10, 1, 0},
             {30, 19, 29, 17, 1, 0}, {20, 27, 19, 24, 1, 0}, {28, 20, 26, 24, 1, 0}, {25, 9, 24, 9, 1, 0},
             {27, 4, 24, 6, 1, 0}, {23, 21, 22, 19, 1, 0}, {7, 13, 7, 10, 1, -1}, {12, 11, 11, 11, 1, 1},
             {28, 26, 26, 26, 1, 0}, {8, 4, 6, 4, 1, 0}, {15, 30, 15, 28, 1, 0}, {30, 14, 28, 14, 1, 0},
             {17, 7, 17, 5, 1, 0}, {29, 10, 28, 6, 1, 0}, {12, 17, 11, 17, 1, 1}, {16, 3, 16, 1, 1, 0},
             {21, 3, 19, 3, 1, 1}, {12, 30, 11, 28, 1, 1}, {18, 16, 18, 15, 1, 0}, {8, 18, 7, 20, 1, -1},
             {5, 4, 1, 1, 1, -1}, {3, 27, 1, 30, 1, -1}, {26, 4, 26, 1, 1, 0}, {5, 21, 2, 20, 1, -1},
             {14, 1, 13, 3, 1, 1}, {30, 9, 28, 8, 1, 0}, {13, 15, 12, 12, 1, 1}, {7, 23, 6, 25, 1, -1}};
    }
    else if(n_bits == SIZE_256_BITS)
    {
        // Pre-trained parameters of BEBLID-256 trained in Liberty data set with
        // a million of patch pairs, 20% positives and 80% negatives
        wl_params_ = {{26, 20, 14, 16, 5, 16}, {17, 17, 15, 15, 2, 7}, {18, 16, 8, 13, 3, 18}, {19, 15, 13, 14, 3, 17},
                      {16, 16, 5, 15, 4, 10}, {25, 10, 16, 16, 6, 11}, {16, 15, 12, 15, 1, 12}, {18, 17, 14, 17, 1, 13},
                      {15, 14, 5, 21, 5, 6}, {14, 14, 11, 7, 4, 2}, {23, 27, 16, 17, 4, 8}, {12, 17, 10, 24, 5, 0},
                      {15, 15, 13, 14, 1, 6}, {16, 16, 14, 16, 1, 7}, {19, 18, 16, 15, 1, 6}, {24, 7, 19, 15, 6, 4},
                      {15, 16, 6, 8, 5, 6}, {24, 16, 8, 15, 7, 22}, {15, 6, 13, 16, 4, 6}, {17, 19, 15, 15, 1, 6},
                      {17, 12, 16, 16, 1, 2}, {11, 15, 7, 25, 6, 0}, {15, 15, 14, 10, 2, 2}, {26, 15, 18, 17, 4, 6},
                      {18, 12, 17, 27, 4, 3}, {9, 15, 6, 8, 6, 1}, {15, 17, 14, 23, 3, 1}, {11, 17, 4, 14, 4, 1},
                      {22, 18, 19, 5, 5, 5}, {11, 18, 11, 5, 5, 3}, {22, 5, 19, 19, 5, 2}, {12, 26, 6, 15, 3, 5},
                      {16, 16, 14, 18, 1, 7}, {22, 26, 22, 13, 5, 2}, {18, 13, 16, 16, 1, 4}, {14, 26, 13, 10, 5, 3},
                      {17, 13, 14, 14, 1, 10}, {21, 16, 19, 7, 3, 4}, {14, 15, 14, 13, 1, 0}, {26, 26, 20, 18, 5, 1},
                      {12, 10, 8, 21, 4, 3}, {14, 17, 13, 7, 3, 0}, {13, 12, 10, 19, 2, 4}, {17, 20, 17, 13, 2, 0},
                      {8, 25, 6, 11, 6, 2}, {27, 11, 20, 24, 4, 3}, {14, 18, 12, 14, 2, 5}, {22, 19, 18, 20, 2, 5},
                      {18, 4, 17, 14, 3, 1}, {13, 28, 13, 18, 3, 3}, {15, 12, 14, 17, 1, 4}, {13, 20, 10, 11, 2, 3},
                      {10, 5, 4, 17, 4, 2}, {7, 18, 3, 18, 3, 2}, {21, 11, 15, 2, 2, 11}, {20, 15, 17, 17, 1, 6},
                      {10, 20, 4, 27, 4, 3}, {24, 25, 23, 7, 6, 0}, {18, 15, 18, 12, 2, 0}, {17, 16, 16, 13, 1, 3},
                      {14, 20, 14, 15, 1, 1}, {17, 17, 17, 14, 1, 0}, {7, 15, 6, 5, 5, 3}, {11, 21, 11, 13, 2, 1},
                      {18, 16, 15, 9, 1, 7}, {19, 19, 18, 15, 1, 2}, {28, 19, 20, 16, 3, 1}, {14, 16, 11, 10, 1, 3},
                      {22, 13, 19, 14, 1, 2}, {9, 10, 4, 4, 4, 3}, {20, 26, 10, 29, 2, 12}, {14, 17, 12, 19, 1, 3},
                      {21, 18, 18, 24, 2, 6}, {16, 15, 15, 19, 1, 4}, {27, 4, 24, 15, 4, 2}, {15, 22, 14, 6, 2, 2},
                      {13, 16, 9, 12, 1, 2}, {12, 12, 11, 18, 1, 2}, {22, 17, 20, 11, 2, 2}, {18, 28, 17, 23, 3, 1},
                      {6, 9, 5, 21, 4, 0}, {12, 3, 8, 11, 3, 5}, {21, 16, 19, 16, 1, 2}, {18, 16, 17, 19, 1, 2},
                      {27, 12, 22, 3, 3, 2}, {13, 27, 4, 26, 4, 3}, {5, 22, 3, 26, 3, 2}, {24, 28, 23, 20, 3, 2},
                      {11, 17, 8, 19, 2, 0}, {13, 16, 11, 16, 1, 3}, {18, 15, 18, 8, 2, 1}, {15, 17, 14, 14, 1, 3},
                      {19, 14, 17, 12, 1, 4}, {25, 10, 22, 20, 2, 0}, {14, 12, 13, 9, 1, 1}, {9, 10, 3, 9, 3, 2},
                      {20, 22, 19, 17, 1, 0}, {16, 24, 16, 10, 2, 0}, {15, 23, 13, 29, 2, 2}, {15, 20, 14, 17, 1, 4},
                      {27, 27, 22, 27, 4, 1}, {14, 7, 6, 3, 3, 3}, {21, 3, 20, 7, 3, 0}, {29, 5, 25, 11, 2, 1},
                      {15, 21, 15, 20, 1, 0}, {8, 17, 8, 11, 2, 1}, {17, 13, 17, 8, 1, 0}, {7, 25, 3, 21, 3, 0},
                      {7, 11, 7, 8, 3, 1}, {4, 11, 3, 26, 3, 2}, {15, 18, 15, 11, 1, 1}, {23, 15, 20, 19, 2, 2},
                      {5, 9, 3, 4, 3, 2}, {28, 18, 25, 8, 3, 0}, {20, 22, 17, 30, 1, 5}, {29, 29, 28, 16, 2, 1},
                      {28, 11, 24, 15, 2, 1}, {20, 7, 18, 9, 1, 2}, {19, 12, 18, 16, 1, 2}, {11, 20, 11, 17, 2, 1},
                      {13, 16, 13, 13, 1, 0}, {29, 3, 23, 5, 2, 0}, {19, 21, 17, 18, 1, 3}, {12, 8, 12, 3, 2, 2},
                      {14, 13, 13, 20, 1, 2}, {11, 21, 9, 29, 2, 3}, {7, 30, 6, 22, 1, 2}, {11, 9, 10, 15, 1, 3},
                      {8, 3, 2, 9, 2, 0}, {19, 7, 18, 3, 3, 2}, {21, 9, 19, 11, 1, 1}, {18, 10, 17, 13, 1, 2},
                      {6, 17, 1, 30, 1, 6}, {17, 29, 16, 28, 2, 1}, {17, 20, 17, 18, 1, 0}, {15, 9, 13, 23, 1, 4},
                      {12, 14, 11, 16, 1, 1}, {7, 17, 5, 14, 2, 1}, {30, 30, 23, 12, 1, 2}, {29, 18, 26, 20, 2, 0},
                      {10, 20, 9, 17, 2, 1}, {4, 15, 2, 8, 2, 2}, {7, 7, 7, 3, 3, 1}, {9, 19, 8, 24, 1, 2},
                      {28, 25, 27, 25, 3, 0}, {13, 15, 12, 18, 1, 1}, {25, 2, 19, 5, 2, 2}, {15, 4, 15, 3, 3, 0},
                      {25, 19, 24, 29, 2, 2}, {18, 24, 18, 20, 1, 1}, {4, 10, 1, 2, 1, 3}, {5, 18, 1, 18, 1, 2},
                      {13, 22, 13, 19, 1, 1}, {10, 26, 8, 28, 2, 0}, {24, 13, 24, 6, 1, 1}, {15, 19, 14, 15, 1, 4},
                      {5, 8, 2, 16, 2, 0}, {12, 4, 11, 2, 2, 0}, {14, 29, 14, 24, 1, 1}, {3, 20, 1, 22, 1, 1},
                      {17, 5, 12, 1, 1, 5}, {21, 16, 20, 23, 1, 2}, {25, 17, 22, 13, 1, 0}, {6, 21, 5, 16, 1, 0},
                      {7, 15, 6, 19, 1, 1}, {20, 17, 19, 15, 1, 1}, {3, 29, 3, 23, 2, 1}, {16, 25, 16, 22, 1, 0},
                      {28, 20, 28, 12, 3, 0}, {27, 13, 23, 10, 1, 0}, {24, 24, 17, 29, 1, 5}, {13, 2, 11, 4, 1, 2},
                      {22, 23, 21, 21, 1, 0}, {19, 30, 19, 24, 1, 1}, {30, 30, 26, 27, 1, 0}, {17, 5, 17, 1, 1, 0},
                      {26, 7, 24, 1, 1, 1}, {28, 6, 28, 3, 3, 0}, {3, 15, 1, 13, 1, 1}, {7, 8, 5, 6, 1, 1},
                      {19, 16, 19, 15, 1, 0}, {12, 9, 11, 7, 1, 0}, {17, 22, 16, 20, 1, 2}, {12, 14, 12, 11, 1, 1},
                      {25, 29, 23, 26, 1, 0}, {15, 19, 15, 18, 1, 0}, {13, 22, 12, 25, 1, 0}, {1, 22, 1, 11, 1, 0},
                      {14, 12, 14, 9, 1, 1}, {10, 27, 9, 23, 1, 2}, {9, 4, 6, 1, 1, 1}, {22, 12, 21, 16, 1, 0},
                      {5, 27, 1, 28, 1, 1}, {30, 14, 28, 7, 1, 0}, {17, 9, 16, 21, 1, 2}, {17, 9, 17, 6, 1, 0},
                      {4, 4, 1, 1, 1, 1}, {30, 2, 28, 5, 1, 0}, {18, 4, 17, 7, 1, 1}, {15, 13, 15, 10, 1, 1},
                      {12, 30, 11, 26, 1, 2}, {16, 28, 15, 29, 1, 1}, {30, 11, 28, 11, 1, 0}, {9, 12, 8, 10, 1, 1},
                      {22, 19, 21, 16, 1, 0}, {30, 20, 29, 26, 1, 0}, {22, 10, 20, 7, 1, 2}, {2, 2, 1, 5, 1, 0},
                      {9, 9, 7, 9, 1, 0}, {27, 1, 25, 3, 1, 0}, {21, 23, 20, 25, 1, 1}, {10, 3, 8, 5, 1, 1},
                      {24, 1, 23, 3, 1, 0}, {5, 29, 4, 28, 1, 0}, {27, 23, 26, 18, 1, 1}, {22, 2, 22, 1, 1, 0},
                      {7, 20, 6, 19, 1, 0}, {12, 26, 9, 25, 1, 2}, {7, 1, 5, 2, 1, 0}, {2, 21, 1, 18, 1, 0},
                      {2, 24, 1, 21, 1, 0}, {8, 17, 8, 14, 1, 0}, {30, 1, 28, 2, 1, 0}, {15, 30, 15, 28, 1, 0},
                      {2, 5, 1, 9, 1, 0}, {18, 28, 17, 26, 1, 1}, {7, 29, 1, 30, 1, 1}, {17, 2, 17, 1, 1, 0},
                      {21, 13, 21, 9, 1, 1}, {29, 15, 27, 15, 1, 0}, {28, 8, 27, 7, 2, 0}, {29, 14, 28, 18, 1, 0},
                      {2, 26, 1, 30, 1, 1}, {16, 8, 16, 6, 1, 0}, {30, 26, 26, 24, 1, 0}, {15, 17, 15, 16, 6, 0},
                      {30, 29, 27, 30, 1, 0}, {3, 30, 1, 28, 1, 0}, {17, 1, 16, 2, 1, 1}, {14, 30, 12, 30, 1, 1},
                      {12, 17, 12, 16, 1, 0}, {4, 18, 4, 16, 1, 0}, {11, 4, 11, 1, 1, 1}, {21, 2, 18, 1, 1, 2},
                      {16, 17, 16, 15, 5, 0}, {3, 1, 2, 2, 1, 0}, {23, 17, 23, 16, 1, 0}, {18, 12, 18, 11, 1, 0},
                      {10, 28, 8, 30, 1, 0}, {12, 10, 12, 8, 1, 1}, {2, 14, 1, 9, 1, 1}, {6, 25, 6, 21, 1, 1},
                      {6, 2, 2, 1, 1, 1}, {30, 19, 29, 20, 1, 0}, {25, 21, 23, 20, 1, 0}, {16, 10, 16, 9, 1, 0},};
    }
    else
    {
        CV_Error(Error::StsBadArg, "n_wls should be either SIZE_512_BITS or SIZE_256_BITS");
    }
}

// Internal function that implements the core of BEBLID descriptor
void BEBLID_Impl::computeBEBLID(const cv::Mat &integralImg,
                                const std::vector<cv::KeyPoint> &keypoints,
                                cv::Mat &descriptors)
{
    CV_DbgAssert(!integralImg.empty());
    CV_DbgAssert(size_t(descriptors.rows) == keypoints.size());
    const int *integralPtr = integralImg.ptr<int>();
    cv::Size frameSize(integralImg.cols - 1, integralImg.rows - 1);

    // Parallel Loop to process descriptors
#ifndef CV_BEBLID_PARALLEL
    const cv::Range range(0, keypoints.size());
#else
    cv::parallel_for_(cv::Range(0, int(keypoints.size())), [&](const Range &range)
    {
#endif
        // Get a pointer to the first element in the range
        ABWLParams *wl;
        float responseFun;
        int areaResponseFun, kpIdx;
        size_t wlIdx;
        int box1x1, box1y1, box1x2, box1y2, box2x1, box2y1, box2x2, box2y2, bit_idx, side;
        uchar byte = 0;
        std::vector<ABWLParams> imgWLParams(wl_params_.size());
        uchar *d = &descriptors.at<uchar>(range.start, 0);

        for (kpIdx = range.start; kpIdx < range.end; kpIdx++)
        {
            // Rectify the weak learners coordinates using the keypoint information
            rectifyABWL(wl_params_, imgWLParams, keypoints[kpIdx], scale_factor_, patch_size_);
            if (isKeypointInTheBorder(keypoints[kpIdx], frameSize, patch_size_, scale_factor_))
            {
                // Code to process the keypoints in the image margins
                for (wlIdx = 0; wlIdx < wl_params_.size(); wlIdx++) {
                    bit_idx = 7 - int(wlIdx % 8);
                    responseFun = computeABWLResponse(imgWLParams[wlIdx], integralImg);
                    // Set the bit to 1 if the response function is less or equal to the threshod
                    byte |= (responseFun <= wl_params_[wlIdx].th) << bit_idx;
                    // If we filled the byte, save it
                    if (bit_idx == 0) {
                        *d = byte;
                        byte = 0;
                        d++;
                    }
                }
            }
            else
            {
                // Code to process the keypoints in the image center
                wl = imgWLParams.data();
                for (wlIdx = 0; wlIdx < wl_params_.size(); wlIdx++)
                {
                    bit_idx = 7 - int(wlIdx % 8);

                    // For the first box, we calculate its margin coordinates
                    box1x1 = wl->x1 - wl->boxRadius;
                    box1y1 = (wl->y1 - wl->boxRadius) * integralImg.cols;
                    box1x2 = wl->x1 + wl->boxRadius + 1;
                    box1y2 = (wl->y1 + wl->boxRadius + 1) * integralImg.cols;
                    // For the second box, we calculate its margin coordinates
                    box2x1 = wl->x2 - wl->boxRadius;
                    box2y1 = (wl->y2 - wl->boxRadius) * integralImg.cols;
                    box2x2 = wl->x2 + wl->boxRadius + 1;
                    box2y2 = (wl->y2 + wl->boxRadius + 1) * integralImg.cols;
                    side = 1 + (wl->boxRadius << 1);

                    // Get the difference between the average level of the two boxes
                    areaResponseFun = (integralPtr[box1y1 + box1x1]  // A of Box1
                        + integralPtr[box1y2 + box1x2]               // D of Box1
                        - integralPtr[box1y1 + box1x2]               // B of Box1
                        - integralPtr[box1y2 + box1x1]               // C of Box1
                        - integralPtr[box2y1 + box2x1]               // A of Box2
                        - integralPtr[box2y2 + box2x2]               // D of Box2
                        + integralPtr[box2y1 + box2x2]               // B of Box2
                        + integralPtr[box2y2 + box2x1]);             // C of Box2

                    // Set the bit to 1 if the response function is less or equal to the threshod
                    byte |= (areaResponseFun <= (wl_params_[wlIdx].th * (side * side))) << bit_idx;
                    wl++;
                    // If we filled the byte, save it
                    if (bit_idx == 0)
                    {
                        *d = byte;
                        byte = 0;
                        d++;
                    }
                }  // End of for each dimension
            }  // End of else (of pixels in the image center)
        }  // End of for each keypoint
#ifdef CV_BEBLID_PARALLEL
    });
#endif
}

Ptr<BEBLID> BEBLID::create(float scale_factor, int n_bits)
{
    return makePtr<BEBLID_Impl>(scale_factor, n_bits);
}
} // END NAMESPACE XFEATURES2D
} // END NAMESPACE CV
