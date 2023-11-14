/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this
 license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without
 modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright
 notice,
 //     this list of conditions and the following disclaimer in the
 documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote
 products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is"
 and
 // any express or implied warranties, including, but not limited to, the
 implied
 // warranties of merchantability and fitness for a particular purpose are
 disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any
 direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"

namespace cv {
namespace structured_light {
class CV_EXPORTS_W SinusCompleGrayCodePattern_Impl CV_FINAL
    : public SinusCompleGrayCodePattern {
  public:
    // Constructor
    explicit SinusCompleGrayCodePattern_Impl(
        const SinusCompleGrayCodePattern::Params &parameters =
            SinusCompleGrayCodePattern::Params());
    // Destructor
    virtual ~SinusCompleGrayCodePattern_Impl() CV_OVERRIDE{};

    // Generate sinusoidal and complementary graycode patterns
    bool generate(OutputArrayOfArrays patternImages) CV_OVERRIDE;

    // decode patterns and compute disparitymap.
    bool decode(const std::vector<std::vector<Mat>> &patternImages,
                OutputArray disparityMap,
                InputArrayOfArrays blackImages = noArray(),
                InputArrayOfArrays whiteImages = noArray(),
                int flags = 0) const CV_OVERRIDE;

    // Compute a confidence map from sinusoidal patterns
    void computeConfidenceMap(InputArrayOfArrays patternImages,
                              OutputArray confidenceMap) const CV_OVERRIDE;

    // Compute a wrapped phase map from the sinusoidal patterns
    void computePhaseMap(InputArrayOfArrays patternImages,
                         OutputArray wrappedPhaseMap) const CV_OVERRIDE;

    // Compute a floor map from complementary graycode patterns and
    // wrappedPhaseMap.
    void computeFloorMap(InputArrayOfArrays patternImages,
                         InputArray confidenceMap, InputArray wrappedPhaseMap,
                         OutputArray floorMap) const CV_OVERRIDE;

    // Unwrap the wrapped phase map to remove phase ambiguities
    void unwrapPhaseMap(InputArray wrappedPhaseMap, InputArray floorMap,
                        OutputArray unwrappedPhaseMap,
                        InputArray shadowMask = noArray()) const CV_OVERRIDE;

    // Compute disparity
    void computeDisparity(InputArray lhsUnwrapMap, InputArray rhsUnwrapMap,
                          OutputArray disparityMap) const CV_OVERRIDE;

  private:
    Params params;
};
// Default parameters value
SinusCompleGrayCodePattern_Impl::Params::Params() {
    width = 1280;
    height = 720;
    nbrOfPeriods = 40;
    shiftTime = 4;
    minDisparity = 0;
    maxDisparity = 320;
    horizontal = false;
    confidenceThreshold = 5.f;
    maxCost = 0.1f;
}

SinusCompleGrayCodePattern_Impl::SinusCompleGrayCodePattern_Impl(
    const SinusCompleGrayCodePattern::Params &parameters)
    : params(parameters) {}

void SinusCompleGrayCodePattern_Impl::computeConfidenceMap(
    InputArrayOfArrays patternImages, OutputArray confidenceMap) const {
    const std::vector<Mat> &imgs =
        *static_cast<const std::vector<Mat> *>(patternImages.getObj());

    Mat &confidence = *static_cast<Mat *>(confidenceMap.getObj());

    CV_Assert(imgs.size() >= params.shiftTime);

    confidence = Mat::zeros(imgs[0].size(), CV_32FC1);

    for (int i = 0; i < params.shiftTime; ++i) {
        cv::Mat fltImg;
        imgs[i].convertTo(fltImg, CV_32FC1);
        confidence += fltImg / params.shiftTime;
    }
}

void SinusCompleGrayCodePattern_Impl::computePhaseMap(
    InputArrayOfArrays patternImages, OutputArray wrappedPhaseMap) const {
    const std::vector<Mat> &imgs =
        *static_cast<const std::vector<Mat> *>(patternImages.getObj());
    Mat &wrappedPhase = *static_cast<Mat *>(wrappedPhaseMap.getObj());

    CV_Assert(imgs.size() >= params.shiftTime);

    const int height = imgs[0].rows;
    const int width = imgs[0].cols;
    wrappedPhase = Mat::zeros(height, width, CV_32FC1);

    std::vector<const uchar *> imgsPtrs(params.shiftTime);
    const double shiftVal = CV_2PI / params.shiftTime;
    for (int i = 0; i < height; ++i) {
        auto wrappedPhasePtr = wrappedPhase.ptr<float>(i);

        for (int j = 0; j < params.shiftTime; ++j) {
            imgsPtrs[j] = imgs[j].ptr<uchar>(i);
        }

        for (int j = 0; j < width; ++j) {
            double molecules = 0.f, denominator = 0.f;
            for (int k = 0; k < params.shiftTime; ++k) {
                molecules += imgsPtrs[k][j] * sin(k * shiftVal);
                denominator += imgsPtrs[k][j] * cos(k * shiftVal);
            }

            wrappedPhasePtr[j] = -atan2(molecules, denominator);
        }
    }
}

void SinusCompleGrayCodePattern_Impl::computeFloorMap(
    InputArrayOfArrays patternImages, InputArray confidenceMap,
    InputArray wrappedPhaseMap, OutputArray floorMap) const {
    const std::vector<Mat> &imgs =
        *static_cast<const std::vector<Mat> *>(patternImages.getObj());
    const Mat &confidence = *static_cast<const Mat *>(confidenceMap.getObj());
    const Mat &wrappedPhase =
        *static_cast<const Mat *>(wrappedPhaseMap.getObj());
    Mat &floor = *static_cast<Mat *>(floorMap.getObj());

    CV_Assert(!imgs.empty() && !confidence.empty() && !wrappedPhase.empty());
    CV_Assert(std::pow(2, imgs.size() - params.shiftTime - 1) ==
              params.nbrOfPeriods);

    const int height = imgs[0].rows;
    const int width = imgs[0].cols;
    floor = Mat::zeros(height, width, CV_16UC1);

    const int grayCodeImgsCount = imgs.size() - params.shiftTime;
    std::vector<const uchar *> imgsPtrs(grayCodeImgsCount);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < grayCodeImgsCount; ++j) {
            imgsPtrs[j] = imgs[params.shiftTime + j].ptr<uchar>(i);
        }
        auto confidencePtr = confidence.ptr<float>(i);
        auto wrappedPhasePtr = wrappedPhase.ptr<float>(i);
        auto floorPtr = floor.ptr<uint16_t>(i);
        for (int j = 0; j < width; ++j) {
            int K1 = 0, K2 = 0, tempVal = 0;
            for (int k = 0; k < grayCodeImgsCount; ++k) {
                tempVal ^= imgsPtrs[k][j] > confidencePtr[j];
                K1 += (tempVal && k != grayCodeImgsCount - 1)
                          ? std::pow(2, grayCodeImgsCount - 2 - k)
                          : 0;
                K2 += tempVal ? std::pow(2, grayCodeImgsCount - 1 - k) : 0;
            }

            K2 = std::floorf((K2 + 1) / 2);
            if (wrappedPhasePtr[j] <= -CV_PI / 2) {
                floorPtr[j] = K2;
            } else if (wrappedPhasePtr[j] >= CV_PI / 2) {
                floorPtr[j] = K2 - 1;
            } else {
                floorPtr[j] = K1;
            }
        }
    }
}

void SinusCompleGrayCodePattern_Impl::unwrapPhaseMap(
    InputArray wrappedPhaseMap, InputArray floorMap,
    OutputArray unwrappedPhaseMap, InputArray shadowMask) const {
    const Mat &wrappedPhase =
        *static_cast<const Mat *>(wrappedPhaseMap.getObj());
    const Mat &floor = *static_cast<const Mat *>(floorMap.getObj());
    Mat &unwrappedPhase = *static_cast<Mat *>(unwrappedPhaseMap.getObj());

    CV_Assert(!wrappedPhase.empty() && !floor.empty());

    const int height = wrappedPhase.rows;
    const int width = wrappedPhase.cols;
    unwrappedPhase = Mat::zeros(height, width, CV_32FC1);

    cv::Mat shadow;
    if (!shadowMask.empty()) {
        shadow = *static_cast<Mat *>(shadowMask.getObj());
    }

    for (int i = 0; i < height; ++i) {
        auto wrappedPhasePtr = wrappedPhase.ptr<float>(i);
        auto floorPtr = floor.ptr<uint16_t>(i);
        auto unwrappedPhasePtr = unwrappedPhase.ptr<float>(i);
        const uchar *shadowPtr =
            shadow.empty() ? nullptr : shadow.ptr<uchar>(i);
        for (int j = 0; j < width; ++j) {
            // we add CV_PI to make wrap map to begin with 0.
            if (shadowPtr) {
                if (shadowPtr[j]) {
                    unwrappedPhasePtr[j] =
                        wrappedPhasePtr[j] + CV_2PI * floorPtr[j] + CV_PI;
                } else {
                    unwrappedPhasePtr[j] = 0.f;
                }
            }
        }
    }
}

bool SinusCompleGrayCodePattern_Impl::generate(OutputArrayOfArrays pattern) {
    std::vector<Mat> &imgs = *static_cast<std::vector<Mat> *>(pattern.getObj());
    imgs.clear();
    const int height = params.horizontal ? params.width : params.height;
    const int width = params.horizontal ? params.height : params.width;
    const int pixelsPerPeriod = width / params.nbrOfPeriods;
    // generate phase-shift imgs.
    for (int i = 0; i < params.shiftTime; ++i) {
        Mat intensityMap = Mat::zeros(height, width, CV_8UC1);
        const float shiftVal = CV_2PI / params.shiftTime * i;

        for (int j = 0; j < height; ++j) {
            auto intensityMapPtr = intensityMap.ptr<uchar>(j);
            for (int k = 0; k < width; ++k) {
                // Set the fringe starting intensity to 0 so that it corresponds
                // to the complementary graycode interval.
                const float wrappedPhaseVal =
                    (k % pixelsPerPeriod) /
                        static_cast<float>(pixelsPerPeriod) * CV_2PI -
                    CV_PI;
                intensityMapPtr[k] =
                    127.5 + 127.5 * cos(wrappedPhaseVal + shiftVal);
            }
        }

        intensityMap = params.horizontal ? intensityMap.t() : intensityMap;
        imgs.push_back(intensityMap);
    }
    // generate complementary graycode imgs.
    const int grayCodeImgsCount = std::log2f(params.nbrOfPeriods) + 1;
    std::vector<uchar> encodeSequential = {0, 255};
    for (int i = 0; i < grayCodeImgsCount; ++i) {
        Mat intensityMap = Mat::zeros(height, width, CV_8UC1);
        const int pixelsPerBlock = width / encodeSequential.size();
        for (int j = 0; j < encodeSequential.size(); ++j) {
            intensityMap(Rect(j * pixelsPerBlock, 0, pixelsPerBlock, height)) =
                encodeSequential[j];
        }

        const int lastSequentialSize = encodeSequential.size();
        for (int j = lastSequentialSize - 1; j >= 0; --j) {
            encodeSequential.push_back(encodeSequential[j]);
        }

        intensityMap = params.horizontal ? intensityMap.t() : intensityMap;
        imgs.push_back(intensityMap);
    }

    return true;
}

void SinusCompleGrayCodePattern_Impl::computeDisparity(
    InputArray leftUnwrapMap, InputArray rightUnwrapMap,
    OutputArray disparityMap) const {
    const Mat &leftUnwrap = *static_cast<const Mat *>(leftUnwrapMap.getObj());
    const Mat &rightUnwrap = *static_cast<const Mat *>(rightUnwrapMap.getObj());
    Mat &disparity = *static_cast<Mat *>(disparityMap.getObj());

    CV_Assert(!leftUnwrap.empty() && !rightUnwrap.empty());

    const int height = leftUnwrap.rows;
    const int width = leftUnwrap.cols;
    disparity = cv::Mat::zeros(height, width, CV_32FC1);

    for (int i = 0; i < height; ++i) {
        auto leftUnwrapPtr = leftUnwrap.ptr<float>(i);
        auto rightUnwrapPtr = rightUnwrap.ptr<float>(i);
        auto disparityPtr = disparity.ptr<float>(i);

        for (int j = 0; j < width; ++j) {
            auto leftVal = leftUnwrapPtr[j];

            if (std::abs(leftVal) < 0.001f) {
                continue;
            }

            float minCost = FLT_MAX, bestDisp = 0.f;

            for (int d = params.minDisparity; d < params.maxDisparity; ++d) {
                if (j - d < 0 || j - d > width - 1) {
                    continue;
                }

                const float curCost = std::abs(leftVal - rightUnwrapPtr[j - d]);

                if (curCost < minCost) {
                    minCost = curCost;
                    bestDisp = d;
                }

                if (minCost < params.maxCost) {
                    if (bestDisp == params.minDisparity ||
                        bestDisp == params.maxDisparity) {
                        disparityPtr[j] = bestDisp;
                        continue;
                    }

                    const float preCost = std::abs(
                        leftVal -
                        rightUnwrapPtr[j - (static_cast<int>(bestDisp) - 1)]);
                    const float nextCost = std::abs(
                        leftVal -
                        rightUnwrapPtr[j - (static_cast<int>(bestDisp) + 1)]);
                    const float denom =
                        std::max(1.f, preCost + nextCost - 2 * minCost);

                    disparityPtr[j] =
                        bestDisp + (preCost - nextCost) / (denom * 2.f);
                }
            }
        }
    }
}

bool SinusCompleGrayCodePattern_Impl::decode(
    const std::vector<std::vector<Mat>> &patternImages,
    OutputArray disparityMap, InputArrayOfArrays blackImages,
    InputArrayOfArrays whiteImages, int flags) const {
    CV_UNUSED(blackImages);
    CV_UNUSED(whiteImages);

    CV_Assert(!patternImages.empty());

    Mat &disparity = *static_cast<Mat *>(disparityMap.getObj());

    if (flags == SINUSOIDAL_COMPLEMENTARY_GRAY_CODE) {
        cv::Mat leftConfidenceMap, rightConfidenceMap;
        cv::Mat leftWrappedMap, rightWrappedMap;
        cv::Mat leftFloorMap, rightFloorMap;
        cv::Mat leftUnwrapMap, rightUnwrapMap;
        // caculate confidence map
        computeConfidenceMap(patternImages[0], leftConfidenceMap);
        computeConfidenceMap(patternImages[1], rightConfidenceMap);
        // caculate wrapped phase map
        computePhaseMap(patternImages[0], leftWrappedMap);
        computePhaseMap(patternImages[1], rightWrappedMap);
        // caculate floor map
        computeFloorMap(patternImages[0], leftConfidenceMap, leftWrappedMap,
                        leftFloorMap);
        computeFloorMap(patternImages[1], rightConfidenceMap, rightWrappedMap,
                        rightFloorMap);
        // caculate unwrapped map
        unwrapPhaseMap(leftWrappedMap, leftFloorMap, leftUnwrapMap,
                       leftConfidenceMap > params.confidenceThreshold);
        unwrapPhaseMap(rightWrappedMap, rightFloorMap, rightUnwrapMap,
                       rightConfidenceMap > params.confidenceThreshold);
        // caculate disparity map
        computeDisparity(leftUnwrapMap, rightUnwrapMap, disparity);
    }

    return true;
}

Ptr<SinusCompleGrayCodePattern> SinusCompleGrayCodePattern::create(
    const SinusCompleGrayCodePattern::Params &params) {
    return makePtr<SinusCompleGrayCodePattern_Impl>(params);
}
} // namespace structured_light
} // namespace cv
