/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

class MSER_Impl CV_FINAL : public cv::fastcv::FCVMSER
{
public:
    explicit MSER_Impl(cv::Size     imgSize,
                       int numNeighbors,
                       int delta,
                       int minArea,
                       int maxArea,
                       float maxVariation,
                       float minDiversity);

    ~MSER_Impl() CV_OVERRIDE;

    cv::Size getImgSize()      CV_OVERRIDE { return imgSize;      };
    int getNumNeighbors() CV_OVERRIDE { return numNeighbors; };
    int getDelta()        CV_OVERRIDE { return delta;        };
    int getMinArea()      CV_OVERRIDE { return minArea;      };
    int getMaxArea()      CV_OVERRIDE { return maxArea;      };
    float getMaxVariation() CV_OVERRIDE { return maxVariation; };
    float getMinDiversity() CV_OVERRIDE { return minDiversity; };

    void detect(InputArray src, std::vector<std::vector<Point>>& contours) CV_OVERRIDE;
    void detect(InputArray src, std::vector<std::vector<Point>>& contours, std::vector<cv::Rect>& boundingBoxes) CV_OVERRIDE;
    void detect(InputArray src, std::vector<std::vector<Point>>& contours, std::vector<cv::Rect>& boundingBoxes,
                                std::vector<ContourData>& contourData) CV_OVERRIDE;

    void detectRegions(InputArray src,
                       std::vector<std::vector<Point>>& contours,
                       std::vector<cv::Rect>& boundingBoxes,
                       std::vector<ContourData>& contourData,
                       bool useBoundingBoxes = true,
                       bool useContourData = true);

    cv::Size imgSize;
    int numNeighbors;
    int delta;
    int minArea;
    int maxArea;
    float maxVariation;
    float minDiversity;

    void *mserHandle;
};


MSER_Impl::MSER_Impl(cv::Size _imgSize,
                     int _numNeighbors,
                     int _delta,
                     int _minArea,
                     int _maxArea,
                     float _maxVariation,
                     float _minDiversity)
{
    CV_Assert(_imgSize.width > 50);
    CV_Assert(_imgSize.height > 5);

    CV_Assert(_numNeighbors == 4 || _numNeighbors == 8);

    INITIALIZATION_CHECK;

    this->imgSize       = _imgSize;
    this->numNeighbors  = _numNeighbors;
    this->delta         = _delta;
    this->minArea       = _minArea;
    this->maxArea       = _maxArea;
    this->maxVariation  = _maxVariation;
    this->minDiversity  = _minDiversity;

    auto initFunc = (this->numNeighbors == 4) ? fcvMserInit : fcvMserNN8Init;

    if (!initFunc(this->imgSize.width, this->imgSize.height, this->delta, this->minArea, this->maxArea,
                  this->maxVariation, this->minDiversity, &this->mserHandle))
    {
        CV_Error(cv::Error::StsInternal, "Failed to initialize MSER");
    }
}


MSER_Impl::~MSER_Impl()
{
    fcvMserRelease(mserHandle);
}


void MSER_Impl::detectRegions(InputArray _src, std::vector<std::vector<Point>>& contours, std::vector<cv::Rect>& boundingBoxes,
                              std::vector<ContourData>& contourData, bool useBoundingBoxes, bool useContourData)
{
    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);
    CV_Assert(_src.size() == this->imgSize);

    Mat src = _src.getMat();

    bool usePointsArray = (this->numNeighbors == 8);

    //bufSize for pts and bboxes
    const uint32_t maxContours = 16384;
    uint32_t numContours;
    std::vector<uint32_t> numPointsInContour(maxContours);

    std::vector<uint16_t> rectArray;
    rectArray.resize(4 * maxContours); // xMin, xMax, yMax, yMin

    uint32_t pointsArraySize = src.total() * 30; // Recommended typical size
    std::vector<uint16_t> pointsArray;
    std::vector<uint32_t> contourStartingPoints;
    uint32_t pathArraySize = src.total() * 4; // Recommended size
    std::vector<uint16_t> pathArray;
    if (usePointsArray)
    {
        pointsArray.resize(pointsArraySize);
    }
    else
    {
        contourStartingPoints.resize(maxContours);
        pathArray.resize(pathArraySize);
    }

    std::vector<uint32_t> contourVariation(maxContours), contourNodeId(maxContours), contourNodeCounter(maxContours);
    std::vector<int8_t> contourPolarity(maxContours);

    int mserRetcode = -1;
    if (this->numNeighbors == 4)
    {
        mserRetcode = fcvMserExtu8_v3(mserHandle, src.data, src.cols, src.rows, src.step,
                                      maxContours, &numContours,
                                      rectArray.data(),
                                      contourStartingPoints.data(),
                                      numPointsInContour.data(),
                                      pathArraySize, pathArray.data(),
                                      contourVariation.data(), contourPolarity.data(), contourNodeId.data(), contourNodeCounter.data());
        CV_LOG_INFO(NULL, "fcvMserExtu8_v3");
    }
    else
    {
        if (useContourData)
        {
            mserRetcode = fcvMserExtNN8u8(mserHandle, src.data, src.cols, src.rows, src.step,
                                          maxContours, &numContours,
                                          rectArray.data(),
                                          numPointsInContour.data(), pointsArraySize, pointsArray.data(),
                                          contourVariation.data(), contourPolarity.data(), contourNodeId.data(), contourNodeCounter.data());
            CV_LOG_INFO(NULL, "fcvMserExtNN8u8");
        }
        else
        {
            mserRetcode = fcvMserNN8u8(mserHandle, src.data, src.cols, src.rows, src.step,
                                       maxContours, &numContours,
                                       rectArray.data(),
                                       numPointsInContour.data(), pointsArraySize, pointsArray.data());
            CV_LOG_INFO(NULL, "fcvMserNN8u8");
        }
    }

    if (mserRetcode != 1)
    {
        CV_Error(cv::Error::StsInternal, "Failed to run MSER");
    }

    contours.clear();
    contours.reserve(numContours);
    if (useBoundingBoxes)
    {
        boundingBoxes.clear();
        boundingBoxes.reserve(numContours);
    }
    if (useContourData)
    {
        contourData.clear();
        contourData.reserve(numContours);
    }
    int ptCtr = 0;
    for (uint32_t i = 0; i < numContours; i++)
    {
        std::vector<Point> contour;
        contour.reserve(numPointsInContour[i]);
        for (uint32_t j = 0; j < numPointsInContour[i]; j++)
        {
            Point pt;
            if (usePointsArray)
            {
                uint32_t idx = (ptCtr + j) * 2;
                pt = Point {pointsArray[idx + 0], pointsArray[idx + 1]};
            }
            else
            {
                uint32_t idx = contourStartingPoints[i] + j * 2;
                pt = Point {pathArray[idx + 0], pathArray[idx + 1]};
            }
            contour.push_back(pt);
        }
        contours.push_back(contour);
        ptCtr += numPointsInContour[i];

        if (useBoundingBoxes)
        {
            uint16_t xMin = rectArray[i * 4 + 0];
            uint16_t xMax = rectArray[i * 4 + 1];
            uint16_t yMax = rectArray[i * 4 + 2];
            uint16_t yMin = rectArray[i * 4 + 3];
            // +1 is because max limit in cv::Rect() is exclusive
            cv::Rect bbox(Point {xMin, yMin},
                          Point {xMax + 1, yMax + 1});
            boundingBoxes.push_back(bbox);
        }

        if (useContourData)
        {
            ContourData data;
            data.variation   = contourVariation[i];
            data.polarity    = contourPolarity[i];
            data.nodeId      = contourNodeId[i];
            data.nodeCounter = contourNodeCounter[i];
            contourData.push_back(data);
        }
    }
}

void MSER_Impl::detect(InputArray src, std::vector<std::vector<Point>> &contours)
{
    std::vector<cv::Rect> boundingBoxes;
    std::vector<ContourData> contourData;
    this->detectRegions(src, contours, boundingBoxes, contourData, /*useBoundingBoxes*/ false, /*useContourData*/ false);
}

void MSER_Impl::detect(InputArray src, std::vector<std::vector<Point>>& contours, std::vector<cv::Rect>& boundingBoxes)
{
    std::vector<ContourData> contourData;
    this->detectRegions(src, contours, boundingBoxes, contourData, /*useBoundingBoxes*/ true, /*useContourData*/ false);
}

void MSER_Impl::detect(InputArray src, std::vector<std::vector<Point>>& contours, std::vector<cv::Rect>& boundingBoxes,
                       std::vector<ContourData>& contourData)
{
    this->detectRegions(src, contours, boundingBoxes, contourData, /*useBoundingBoxes*/ true, /*useContourData*/ true);
}

Ptr<FCVMSER> FCVMSER::create(const cv::Size& imgSize,
                             int numNeighbors,
                             int delta,
                             int minArea,
                             int maxArea,
                             float maxVariation,
                             float minDiversity)
{
    CV_Assert(numNeighbors > 0 && delta >= 0 && minArea >= 0 && maxArea >= 0);
    return makePtr<MSER_Impl>(imgSize, numNeighbors, delta, minArea, maxArea, maxVariation, minDiversity);
}

} // fastcv::
} // cv::
