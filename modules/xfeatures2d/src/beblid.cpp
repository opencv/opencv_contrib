// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Author: Iago Suarez <iago.suarez.canosa@alumnos.upm.es>

// Implementation of the article:
//     Iago Suarez, Ghesn Sfeir, Jose M. Buenaposada, and Luis Baumela.
//     BEBLID: Boosted Efficient Binary Local Image Descriptor.
//     Pattern Recognition Letters, 133:366–372, 2020.

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
// Same as previous with floating point threshold
struct ABWLParamsFloatTh
{
    int x1, y1, x2, y2, boxRadius;
    float th;
};

// BEBLID implementation
template <class WeakLearnerT>
class BEBLID_Impl CV_FINAL: public BEBLID
{
public:

    // constructor
    explicit BEBLID_Impl(float scale_factor, const std::vector<WeakLearnerT>& wl_params);

    // destructor
    ~BEBLID_Impl() CV_OVERRIDE = default;

    // returns the descriptor length in bytes
    int descriptorSize() const CV_OVERRIDE { return int(wl_params_.size() / 8); }

    // returns the descriptor type
    int descriptorType() const CV_OVERRIDE { return CV_8UC1; }

    // returns the default norm type
    int defaultNorm() const CV_OVERRIDE { return cv::NORM_HAMMING;  }

    void setScaleFactor(float scale_factor) CV_OVERRIDE { scale_factor_ = scale_factor; }
    float getScaleFactor() const { return scale_factor_;}

    // compute descriptors given keypoints
    void compute(InputArray image, vector<KeyPoint> &keypoints, OutputArray descriptors) CV_OVERRIDE;

private:
    std::vector<WeakLearnerT> wl_params_;
    float scale_factor_;
    cv::Size patch_size_;

    void computeBoxDiffsDescriptor(const cv::Mat &integralImg,
                                   const std::vector<cv::KeyPoint> &keypoints,
                                   cv::Mat &descriptors);
}; // END BEBLID_Impl CLASS


// TEBLID implementation
class TEBLID_Impl CV_FINAL: public TEBLID
{
public:

    // constructor
    explicit TEBLID_Impl(float scale_factor, const std::vector<ABWLParamsFloatTh>& wl_params) :
        impl(scale_factor, wl_params){}

    // destructor
    ~TEBLID_Impl() CV_OVERRIDE = default;

    // returns the descriptor length in bytes
    int descriptorSize() const CV_OVERRIDE { return impl.descriptorSize(); }

    // returns the descriptor type
    int descriptorType() const CV_OVERRIDE { return impl.descriptorType(); }

    // returns the default norm type
    int defaultNorm() const CV_OVERRIDE { return impl.defaultNorm();  }

    // compute descriptors given keypoints
    void compute(InputArray image, vector<KeyPoint> &keypoints, OutputArray descriptors) CV_OVERRIDE
    {
        impl.compute(image, keypoints, descriptors);
    }

private:
    BEBLID_Impl<ABWLParamsFloatTh> impl;
}; // END TEBLID_Impl CLASS

Ptr<TEBLID> TEBLID::create(float scale_factor, int n_bits)
{
    if (n_bits == TEBLID::SIZE_512_BITS)
    {
        #include "teblid.p512.hpp"
        return makePtr<TEBLID_Impl>(scale_factor, teblid_wl_params_512);
    }
    else if(n_bits == TEBLID::SIZE_256_BITS)
    {
        #include "teblid.p256.hpp"
        return makePtr<TEBLID_Impl>(scale_factor, teblid_wl_params_256);
    }
    else
    {
        CV_Error(Error::StsBadArg, "n_bits should be either TEBLID::SIZE_512_BITS or TEBLID::SIZE_256_BITS");
    }
}

/**
 * @brief Function that determines if a keypoint is close to the image border.
 * @param kp The detected keypoint
 * @param imgSize The size of the image
 * @param patchSize The size of the normalized patch where the measurement functions were learnt.
 * @param scaleFactor A scale factor that magnifies the measurement functions w.r.t. the keypoint.
 * @return true if the keypoint is in the border, false otherwise
 */
static inline bool isKeypointInTheBorder(const cv::KeyPoint &kp,
                                         const cv::Size &imgSize,
                                         const cv::Size &patchSize = {32, 32},
                                         float scaleFactor = 1)
{
    // This would be the correct measure but since we will compare with half of the size, use this as border size
    float s = scaleFactor * kp.size / (patchSize.width + patchSize.height);
    cv::Size2f border(patchSize.width * s * CV_BEBLID_EXTRA_RATIO_MARGIN,
                      patchSize.height * s * CV_BEBLID_EXTRA_RATIO_MARGIN);

    if (kp.pt.x < border.width || kp.pt.x + border.width >= imgSize.width)
        return true;

    if (kp.pt.y < border.height || kp.pt.y + border.height >= imgSize.height)
        return true;

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
template< typename WeakLearnerT>
static inline void rectifyABWL(const std::vector<WeakLearnerT> &wlPatchParams,
                               std::vector<WeakLearnerT> &wlImageParams,
                               const cv::KeyPoint &kp,
                               float scaleFactor = 1,
                               const cv::Size &patchSize = cv::Size(32, 32))
{
    float m00, m01, m02, m10, m11, m12;
    float s, cosine, sine;

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

    for (size_t i = 0; i < wlPatchParams.size(); i++)
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
template <typename WeakLearnerT>
static inline float computeABWLResponse(const WeakLearnerT &wlImageParams,
                                        const cv::Mat &integralImage)
{
    CV_DbgAssert(!integralImage.empty());
    CV_DbgAssert(integralImage.type() == CV_32SC1);

    int frameWidth, frameHeight, box1x1, box1y1, box1x2, box1y2, box2x1, box2y1, box2x2, box2y2;
    int A, B, C, D;
    int box_area1, box_area2;
    float sum1, sum2, average1, average2;
    // Since the integral image has one extra row and col, calculate the patch dimensions
    frameWidth = integralImage.cols;
    frameHeight = integralImage.rows;

    // For the first box, we calculate its margin coordinates
    box1x1 = wlImageParams.x1 - wlImageParams.boxRadius;
    if (box1x1 < 0)
        box1x1 = 0;
    else if (box1x1 >= frameWidth - 1)
        box1x1 = frameWidth - 2;
    box1y1 = wlImageParams.y1 - wlImageParams.boxRadius;
    if (box1y1 < 0)
        box1y1 = 0;
    else if (box1y1 >= frameHeight - 1)
        box1y1 = frameHeight - 2;
    box1x2 = wlImageParams.x1 + wlImageParams.boxRadius + 1;
    if (box1x2 <= 0)
        box1x2 = 1;
    else if (box1x2 >= frameWidth)
        box1x2 = frameWidth - 1;
    box1y2 = wlImageParams.y1 + wlImageParams.boxRadius + 1;
    if (box1y2 <= 0)
        box1y2 = 1;
    else if (box1y2 >= frameHeight)
        box1y2 = frameHeight - 1;
    CV_DbgAssert((box1x1 < box1x2 && box1y1 < box1y2) && "Box 1 has size 0");

    // For the second box, we calculate its margin coordinates
    box2x1 = wlImageParams.x2 - wlImageParams.boxRadius;
    if (box2x1 < 0)
        box2x1 = 0;
    else if (box2x1 >= frameWidth - 1)
        box2x1 = frameWidth - 2;
    box2y1 = wlImageParams.y2 - wlImageParams.boxRadius;
    if (box2y1 < 0)
        box2y1 = 0;
    else if (box2y1 >= frameHeight - 1)
        box2y1 = frameHeight - 2;
    box2x2 = wlImageParams.x2 + wlImageParams.boxRadius + 1;
    if (box2x2 <= 0)
        box2x2 = 1;
    else if (box2x2 >= frameWidth)
        box2x2 = frameWidth - 1;
    box2y2 = wlImageParams.y2 + wlImageParams.boxRadius + 1;
    if (box2y2 <= 0)
        box2y2 = 1;
    else if (box2y2 >= frameHeight)
        box2y2 = frameHeight - 1;
    CV_DbgAssert((box2x1 < box2x2 && box2y1 < box2y2) && "Box 2 has size 0");

    // Read the integral image values for the first box
    A = integralImage.at<int>(box1y1, box1x1);
    B = integralImage.at<int>(box1y1, box1x2);
    C = integralImage.at<int>(box1y2, box1x1);
    D = integralImage.at<int>(box1y2, box1x2);

    // Calculate the mean intensity value of the pixels in the box
    sum1 = float(A + D - B - C);
    box_area1 = (box1y2 - box1y1) * (box1x2 - box1x1);
    CV_DbgAssert(box_area1 > 0);
    average1 = sum1 / box_area1;

    // Calculate the indices on the integral image where the box falls
    A = integralImage.at<int>(box2y1, box2x1);
    B = integralImage.at<int>(box2y1, box2x2);
    C = integralImage.at<int>(box2y2, box2x1);
    D = integralImage.at<int>(box2y2, box2x2);

    // Calculate the mean intensity value of the pixels in the box
    sum2 = float(A + D - B - C);
    box_area2 = (box2y2 - box2y1) * (box2x2 - box2x1);
    CV_DbgAssert(box_area2 > 0);
    average2 = sum2 / box_area2;

    return average1 - average2;
}

// descriptor computation using keypoints
template <class WeakLearnerT>
void BEBLID_Impl<WeakLearnerT>::compute(InputArray _image, vector<KeyPoint> &keypoints, OutputArray _descriptors)
{
    Mat image = _image.getMat();

    if (image.empty())
        return;

    if (keypoints.empty())
    {
        // clean output buffer (it may be reused with "allocated" data)
        _descriptors.release();
        return;
    }

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
    computeBoxDiffsDescriptor(integralImg, keypoints, descriptors);
}

// constructor
template <class WeakLearnerT>
BEBLID_Impl<WeakLearnerT>::BEBLID_Impl(float scale_factor, const std::vector<WeakLearnerT>& wl_params)
    :  wl_params_(wl_params), scale_factor_(scale_factor), patch_size_(32, 32)
{
}

// Internal function that implements the core of BEBLID descriptor
template<class WeakLearnerT>
void BEBLID_Impl<WeakLearnerT>::computeBoxDiffsDescriptor(const cv::Mat &integralImg,
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
#endif
    {
        // Get a pointer to the first element in the range
        WeakLearnerT *wl;
        float responseFun;
        int areaResponseFun, kpIdx;
        size_t wlIdx;
        int box1x1, box1y1, box1x2, box1y2, box2x1, box2y1, box2x2, box2y2, bit_idx, side;
        uchar byte = 0;
        std::vector<WeakLearnerT> imgWLParams(wl_params_.size());
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
                    if (bit_idx == 0)
                    {
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
    }  // End of thread scope
#ifdef CV_BEBLID_PARALLEL
    );
#endif
}

Ptr<BEBLID> BEBLID::create(float scale_factor, int n_bits)
{
    if (n_bits == BEBLID::SIZE_512_BITS)
    {
        #include "beblid.p512.hpp"
        return makePtr<BEBLID_Impl<ABWLParams>>(scale_factor, beblid_wl_params_512);
    }
    else if(n_bits == BEBLID::SIZE_256_BITS)
    {
        #include "beblid.p256.hpp"
        return makePtr<BEBLID_Impl<ABWLParams>>(scale_factor, beblid_wl_params_256);
    }
    else
    {
        CV_Error(Error::StsBadArg, "n_bits should be either BEBLID::SIZE_512_BITS or BEBLID::SIZE_256_BITS");
    }
}

String BEBLID::getDefaultName() const
{
  return (Feature2D::getDefaultName() + ".BEBLID");
}

String TEBLID::getDefaultName() const
{
  return (Feature2D::getDefaultName() + ".TEBLID");
}

} // END NAMESPACE XFEATURES2D
} // END NAMESPACE CV
