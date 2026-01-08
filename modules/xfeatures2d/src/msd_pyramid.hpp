// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_XFEATURES2D_MSD_PYRAMID_HPP__
#define __OPENCV_XFEATURES2D_MSD_PYRAMID_HPP__

#include "precomp.hpp"

namespace cv
{
namespace xfeatures2d
{
/*!
    MSD Image Pyramid.
 */
class MSDImagePyramid
{
    // Multi-threaded construction of the scale-space pyramid
    struct MSDImagePyramidBuilder : ParallelLoopBody
    {

        MSDImagePyramidBuilder(const cv::Mat& _im, std::vector<cv::Mat>* _m_imPyr, float _scaleFactor)
        {
            im = &_im;
            m_imPyr = _m_imPyr;
            scaleFactor = _scaleFactor;

        }

        void operator()(const Range& range) const CV_OVERRIDE
        {
            for (int lvl = range.start; lvl < range.end; lvl++)
            {
                float scale = 1 / std::pow(scaleFactor, (float) lvl);
                (*m_imPyr)[lvl] = cv::Mat(cv::Size(cvRound(im->cols * scale), cvRound(im->rows * scale)), im->type());
                cv::resize(*im, (*m_imPyr)[lvl], cv::Size((*m_imPyr)[lvl].cols, (*m_imPyr)[lvl].rows), 0.0, 0.0, cv::INTER_AREA);
            }
        }
        const cv::Mat* im;
        std::vector<cv::Mat>* m_imPyr;
        float scaleFactor;
    };

public:

    MSDImagePyramid(const cv::Mat &im, const int nLevels, const float scaleFactor = 1.6f)
    {
        m_nLevels = nLevels;
        m_scaleFactor = scaleFactor;
        m_imPyr.clear();
        m_imPyr.resize(nLevels);

        m_imPyr[0] = im.clone();

        if (m_nLevels > 1)
        {
            parallel_for_(Range(1, nLevels), MSDImagePyramidBuilder(im, &m_imPyr, scaleFactor));
        }
    }
    ~MSDImagePyramid() {};

    std::vector<cv::Mat> getImPyr() const
    {
        return m_imPyr;
    };

private:

    std::vector<cv::Mat> m_imPyr;
    int m_nLevels;
    float m_scaleFactor;
};
}
}

#endif