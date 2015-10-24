/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


License Agreement
For Open Source Computer Vision Library
(3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

 * Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.

***********************************************************************************
Maximal Self-Dissimilarity (MSD) Interest Point Detector

This is an implementation of the MSD interest point detector
presented in the scientific publication:

[1] F. Tombari, L. Di Stefano
"Interest Points via Maximal Self-Dissimilarities"
12th Asian Conference on Computer Vision (ACCV), 2014

The code is ported from the stand-alone implementation available
at this repository:
www.github.com/fedassa/msdDetector

AUTHORS:  Federico Tombari (fedassa@gmail.com) (original code),
          Daniele De Gregorio (degregorio.daniele@gmail.com) (OpenCV porting)

University of Bologna, Open Perception

 */

#include "precomp.hpp"
#include <limits>

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

                void operator()(const Range& range) const
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

            MSDImagePyramid(const cv::Mat &im, const int nLevels, const float scaleFactor = 1.6f);
            ~MSDImagePyramid();

            const std::vector<cv::Mat> getImPyr() const
            {
                return m_imPyr;
            };

        private:

            std::vector<cv::Mat> m_imPyr;
            int m_nLevels;
            float m_scaleFactor;
        };

        MSDImagePyramid::MSDImagePyramid(const cv::Mat & im, const int nLevels, const float scaleFactor)
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

        MSDImagePyramid::~MSDImagePyramid()
        {
        }

        /*!
            MSD Implementation.
         */
        class MSDDetector_Impl : public MSDDetector
        {
        public:

            // Multi-threaded contextualSelfDissimilarity method
            struct MSDSelfDissimilarityScan : ParallelLoopBody
            {

                MSDSelfDissimilarityScan(MSDDetector_Impl& _detector, std::vector< std::vector<float> >* _saliency, cv::Mat& _img, int _level, int _border, int _split)
                {
                    detector = &_detector;
                    saliency = _saliency;
                    img = &_img;
                    split = _split;
                    level = _level;
                    border = _border;
                    int w = img->cols - border * 2;
                    chunkSize = w / split;
                    remains = w - chunkSize*split;
                }

                void operator()(const Range& range) const
                {
                    for (int i = range.start; i < range.end; i++)
                    {
                        int start = border + i*chunkSize;
                        int end = border + (i + 1) * chunkSize;
                        if (remains > 0)
                            if (i == split - 1)
                            {
                                end = img->cols - border;
                            }
                        detector->contextualSelfDissimilarity(*img, start, end, &saliency->at(level)[0]);
                    }
                }

                MSDDetector_Impl* detector;
                std::vector< std::vector<float> >* saliency;
                cv::Mat* img;
                int level;
                int split;
                int border;
                int chunkSize;
                int remains;
            };

            /**
             * Constructor
             * @param patch_radius Patch radius
             * @param search_area_radius Search Area radius
             * @param nms_radius  Non Maxima Suppression spatial radius
             * @param nms_scale_radius Non Maxima Suppression scale radius
             * @param th_saliency Saliency threshold
             * @param kNN number of nearest neighbors (k)
             * @param scale_factor Scale factor for building up the image pyramid
             * @param n_scales Number of scales number of scales for building up the image pyramid (if set to -1, this number is automatically determined)
             * @param compute_orientation Flag for associating a canoncial orientation to each keypoint
             */
            MSDDetector_Impl(int patch_radius, int search_area_radius,
                    int nms_radius, int nms_scale_radius, float th_saliency, int kNN, float scale_factor,
                    int n_scales, bool compute_orientation)
            : m_patch_radius(patch_radius), m_search_area_radius(search_area_radius), m_nms_radius(nms_radius),
              m_nms_scale_radius(nms_scale_radius), m_th_saliency(th_saliency), m_kNN(kNN), m_scale_factor(scale_factor),
              m_n_scales(n_scales), m_compute_orientation(compute_orientation)

            {
            }

            void detect(InputArray _image, std::vector<KeyPoint>& keypoints, InputArray _mask)
            {
                m_mask = _mask.getMat();

                int border = m_search_area_radius + m_patch_radius;

                cv::Mat img = _image.getMat();
                if (m_n_scales == -1)
                    m_cur_n_scales = cvFloor(std::log(cv::min(img.cols, img.rows) / ((m_patch_radius + m_search_area_radius)*2.0 + 1)) / std::log(m_scale_factor));
                else
                    m_cur_n_scales = m_n_scales;

                cv::Mat imgG;
                if (img.channels() == 1)
                    imgG = img;
                else
                    cv::cvtColor(img, imgG, cv::COLOR_BGR2GRAY);

                MSDImagePyramid scaleSpacer(imgG, m_cur_n_scales, m_scale_factor);
                m_scaleSpace = scaleSpacer.getImPyr();

                keypoints.clear();
                std::vector< std::vector<float> > saliency;
                saliency.resize(m_cur_n_scales);

                for (int r = 0; r < m_cur_n_scales; r++)
                {
                    saliency[r].resize(m_scaleSpace[r].rows * m_scaleSpace[r].cols);
                    fill(saliency[r].begin(), saliency[r].end(), 0.0f);
                }

                for (int r = 0; r < m_cur_n_scales; r++)
                {
                    int steps = cv::getNumThreads();
                    parallel_for_(Range(0, steps), MSDSelfDissimilarityScan((*this), &saliency, m_scaleSpace[r], r, border, steps));
                }

                nonMaximaSuppression(saliency, keypoints);

                for (int r = 0; r < m_cur_n_scales; r++)
                {
                    saliency[r].clear();
                }

                m_scaleSpace.clear();

            }

        protected:

            // Patch radius
            int m_patch_radius;
            // Search area radius
            int m_search_area_radius;
            // Non Maxima Suppression Spatial Radius
            int m_nms_radius;
            // Non Maxima Suppression Scale Radius
            int m_nms_scale_radius;
            //Saliency threshold
            float m_th_saliency;
            //k nearest neighbors
            int m_kNN;
            //Scale factor
            float m_scale_factor;
            //Number of scales
            int m_n_scales;
            //Current number of scales
            int m_cur_n_scales;
            //Compute orientation flag
            bool m_compute_orientation;

        private:


            // Scale-space image pyramid
            std::vector<cv::Mat> m_scaleSpace;
            // Input binary mask
            cv::Mat m_mask;

            /**
             * Computes the normalized average value of input vector
             * @param minVals input vector
             * @param den normalization factor (pre-multiplied by the number of elements of the input vector, assumed constant)
             * @return normalized average value
             */
            inline float computeAvgDistance(std::vector<int> &minVals, int den)
            {
                float avg_dist = 0.0f;
                for (unsigned int i = 0; i < minVals.size(); i++)
                    avg_dist += minVals[i];

                avg_dist /= den;
                return avg_dist;
            }

            /**
             * Computer the Contextual Self-Dissimilarity (CSD, [1]) for a specific range of image pixels (row-wise)
             * @param img input image
             * @param xmin left-most range limit for the image pixels being processed
             * @param xmax right-most range limit for the image pixels being processed
             * @param saliency output array being filled with the CSD value computed at each input pixel
             */
            void contextualSelfDissimilarity(cv::Mat &img, int xmin, int xmax, float* saliency);

            /**
             * Associates a canonical orientation (computed as in [1]) to each extracted key-point
             * @param img input image
             * @param x column index of the key-point on the input image
             * @param y row index of the key-point on the input image
             * @param circle pre-computed LUT used in the function
             * @return angle of the canonical orientation (in radians)
             */
            float computeOrientation(cv::Mat &img, int x, int y, std::vector<cv::Point2f> circle);

            /**
             * Computes the Non-Maxima Suppression (NMS) over the scale-space as in [1] for all elements of the image pyramid
             * @param saliency input saliency associated to each element of the image pyramid
             * @param keypoints key-points obtained as local maxima of the saliency
             */
            void nonMaximaSuppression(std::vector< std::vector<float> > & saliency, std::vector<cv::KeyPoint> & keypoints);

            /**
             * Computes the floating point interpolation of a key-point coordinates
             * @param x column index of the key-point at its scale of the image pyramid
             * @param yrow index of the key-point at its scale of the image pyramid
             * @param scale scale of the key-point over the image pyramid
             * @param saliency pointer to the saliency array
             * @param p_res interpolated coordinates of the key-point referred to the lowest level of the pyramid (i.e. in the ref. frame of the input image)
             * @return false if the current key-point has to be rejected, true otherwise
             */
            bool rescalePoint(int x, int y, int scale, std::vector< std::vector<float> > & saliency, cv::Point2f & p_res);

        };

        bool MSDDetector_Impl::rescalePoint(int i, int j, int scale, std::vector< std::vector<float> > & saliency, cv::Point2f &p_res)
        {

            const float deriv_scale = 0.5f;
            int width_s = m_scaleSpace[scale].cols;
            //const float second_deriv_scale = 1.0f;
            const float cross_deriv_scale = 0.25f;

            cv::Vec2f dD((saliency[scale][j * width_s + i + 1] - saliency[scale][j * width_s + i - 1]) * deriv_scale,
                    (saliency[scale][(j + 1) * width_s + i] - saliency[scale][(j - 1) * width_s + i]) * deriv_scale);

            float cc = saliency[scale][j * width_s + i] * 2;
            float dxx = (saliency[scale][j * width_s + i + 1] + saliency[scale][j * width_s + i - 1] - cc); // * second_deriv_scale;
            float dyy = (saliency[scale][(j + 1) * width_s + i] + saliency[scale][(j - 1) * width_s + i] - cc); // * second_deriv_scale;
            float dxy = (saliency[scale][(j + 1) * width_s + i + 1] - saliency[scale][(j + 1) * width_s + i - 1] -
                    saliency[scale][(j - 1) * width_s + i + 1] + saliency[scale][(j - 1) * width_s + i - 1]) * cross_deriv_scale;

            cv::Matx22f H(dxx, dxy, dxy, dyy);

            cv::Vec2f X;
            cv::solve(H, dD, X, cv::DECOMP_LU);

            float xr = -X[1];
            float xc = -X[0];

            if (std::abs(xr) > 5 || std::abs(xc) > 5)
                return false;

            if (scale == 0)
            {
                p_res.x = i + xc + 0.5f;
                p_res.y = j + xr + 0.5f;
            } else
            {
                float effectiveScaleFactor = std::pow(m_scale_factor, scale);
                p_res.x = (i + xc + 0.5f) * effectiveScaleFactor;
                p_res.y = (j + xr + 0.5f) * effectiveScaleFactor;

                p_res.x -= 0.5f;
                p_res.y -= 0.5f;

                if (p_res.x < 0 || p_res.x >= m_scaleSpace[0].cols || p_res.y < 0 || p_res.y >= m_scaleSpace[0].rows)
                {
                    return false;
                }
            }

            return true;
        }

        void MSDDetector_Impl::contextualSelfDissimilarity(cv::Mat &img, int xmin, int xmax, float* saliency)
        {
            int r_s = m_patch_radius;
            int r_b = m_search_area_radius;
            int k = m_kNN;

            int w = img.cols;
            int h = img.rows;

            int side_s = 2 * r_s + 1;
            int side_b = 2 * r_b + 1;
            int border = r_s + r_b;
            int temp;
            int den = side_s * side_s * k;

            std::vector<int> minVals(k);
            int *acc = new int[side_b * side_b];
            int **vCol = new int *[w];
            for (int i = 0; i < w; i++)
                vCol[i] = new int[side_b * side_b];

            //first position
            int x = xmin;
            int y = border;

            int ctrInd = 0;
            for (int kk = 0; kk < k; kk++)
                minVals[kk] = std::numeric_limits<int>::max();

            for (int j = y - r_b; j <= y + r_b; j++)
            {

                for (int i = x - r_b; i <= x + r_b; i++)
                {
                    if (j == y && i == x)
                        continue;

                    acc[ctrInd] = 0;
                    for (int u = -r_s; u <= r_s; u++)
                    {
                        vCol[x + u][ctrInd] = 0;
                        for (int v = -r_s; v <= r_s; v++)
                        {

                            temp = img.at<unsigned char>(j + v, i + u) - img.at<unsigned char>(y + v, x + u);
                            vCol[x + u][ctrInd] += (temp * temp);
                        }
                        acc[ctrInd] += vCol[x + u][ctrInd];
                    }

                    if (acc[ctrInd] < minVals[k - 1])
                    {
                        minVals[k - 1] = acc[ctrInd];

                        for (int kk = k - 2; kk >= 0; kk--)
                        {
                            if (minVals[kk] > minVals[kk + 1])
                            {
                                std::swap(minVals[kk], minVals[kk + 1]);
                            } else
                                break;
                        }
                    }

                    ctrInd++;
                }
            }
            saliency[y * w + x] = computeAvgDistance(minVals, den);

            for (x = xmin + 1; x < xmax; x++)
            {
                ctrInd = 0;
                for (int kk = 0; kk < k; kk++)
                    minVals[kk] = std::numeric_limits<int>::max();

                for (int j = y - r_b; j <= y + r_b; j++)
                {
                    for (int i = x - r_b; i <= x + r_b; i++)
                    {
                        if (j == y && i == x)
                            continue;

                        vCol[x + r_s][ctrInd] = 0;
                        for (int v = -r_s; v <= r_s; v++)
                        {
                            temp = img.at<unsigned char>(j + v, i + r_s) - img.at<unsigned char>(y + v, x + r_s);
                            vCol[x + r_s][ctrInd] += (temp * temp);
                        }

                        acc[ctrInd] = acc[ctrInd] + vCol[x + r_s][ctrInd] - vCol[x - r_s - 1][ctrInd];

                        if (acc[ctrInd] < minVals[k - 1])
                        {
                            minVals[k - 1] = acc[ctrInd];
                            for (int kk = k - 2; kk >= 0; kk--)
                            {
                                if (minVals[kk] > minVals[kk + 1])
                                {
                                    std::swap(minVals[kk], minVals[kk + 1]);
                                } else
                                    break;
                            }
                        }

                        ctrInd++;
                    }
                }
                saliency[y * w + x] = computeAvgDistance(minVals, den);
            }

            for (y = border + 1; y < h - border; y++)
            {
                ctrInd = 0;
                for (int kk = 0; kk < k; kk++)
                    minVals[kk] = std::numeric_limits<int>::max();
                x = xmin;

                for (int j = y - r_b; j <= y + r_b; j++)
                {
                    for (int i = x - r_b; i <= x + r_b; i++)
                    {
                        if (j == y && i == x)
                            continue;

                        acc[ctrInd] = 0;
                        for (int u = -r_s; u <= r_s; u++)
                        {
                            temp = img.at<unsigned char>(j + r_s, i + u) - img.at<unsigned char>(y + r_s, x + u);
                            vCol[x + u][ctrInd] += (temp * temp);

                            temp = img.at<unsigned char>(j - r_s - 1, i + u) - img.at<unsigned char>(y - r_s - 1, x + u);
                            vCol[x + u][ctrInd] -= (temp * temp);

                            acc[ctrInd] += vCol[x + u][ctrInd];
                        }

                        if (acc[ctrInd] < minVals[k - 1])
                        {
                            minVals[k - 1] = acc[ctrInd];

                            for (int kk = k - 2; kk >= 0; kk--)
                            {
                                if (minVals[kk] > minVals[kk + 1])
                                {
                                    std::swap(minVals[kk], minVals[kk + 1]);
                                } else
                                    break;
                            }
                        }

                        ctrInd++;
                    }
                }
                saliency[y * w + x] = computeAvgDistance(minVals, den);

                for (x = xmin + 1; x < xmax; x++)
                {
                    ctrInd = 0;
                    for (int kk = 0; kk < k; kk++)
                        minVals[kk] = std::numeric_limits<int>::max();

                    for (int j = y - r_b; j <= y + r_b; j++)
                    {
                        for (int i = x - r_b; i <= x + r_b; i++)
                        {
                            if (j == y && i == x)
                                continue;

                            temp = img.at<unsigned char>(j + r_s, i + r_s) - img.at<unsigned char>(y + r_s, x + r_s);
                            vCol[x + r_s][ctrInd] += (temp * temp);

                            temp = img.at<unsigned char>(j - r_s - 1, i + r_s) - img.at<unsigned char>(y - r_s - 1, x + r_s);
                            vCol[x + r_s][ctrInd] -= (temp * temp);

                            acc[ctrInd] = acc[ctrInd] + vCol[x + r_s][ctrInd] - vCol[x - r_s - 1][ctrInd];

                            if (acc[ctrInd] < minVals[k - 1])
                            {
                                minVals[k - 1] = acc[ctrInd];

                                for (int kk = k - 2; kk >= 0; kk--)
                                {
                                    if (minVals[kk] > minVals[kk + 1])
                                    {
                                        std::swap(minVals[kk], minVals[kk + 1]);
                                    } else
                                        break;
                                }
                            }
                            ctrInd++;
                        }
                    }
                    saliency[y * w + x] = computeAvgDistance(minVals, den);
                }
            }

            for (int i = 0; i < w; i++)
                delete[] vCol[i];
            delete[] vCol;
            delete[] acc;
        }

        float MSDDetector_Impl::computeOrientation(cv::Mat &img, int x, int y, std::vector<cv::Point2f> circle)
        {
            int temp;

            int nBins = 36;
            float step = float((2 * CV_PI) / nBins);
            std::vector<float> hist(nBins, 0);
            std::vector<int> dists(circle.size(), 0);

            int minDist = std::numeric_limits<int>::max();
            int maxDist = -1;

            for (int k = 0; k < (int) circle.size(); k++)
            {

                int j = y + static_cast<int> (circle[k].y);
                int i = x + static_cast<int> (circle[k].x);

                for (int v = -m_patch_radius; v <= m_patch_radius; v++)
                {
                    for (int u = -m_patch_radius; u <= m_patch_radius; u++)
                    {
                        temp = img.at<unsigned char>(j + v, i + u) - img.at<unsigned char>(y + v, x + u);
                        dists[k] += temp*temp;
                    }
                }

                if (dists[k] > maxDist)
                    maxDist = dists[k];
                if (dists[k] < minDist)
                    minDist = dists[k];
            }

            float deltaAngle = 0.0f;
            for (int k = 0; k < (int) circle.size(); k++)
            {
                float angle = deltaAngle;
                float weight = (1.0f * maxDist - dists[k]) / (maxDist - minDist);

                float binF;
                if (angle >= 2 * CV_PI)
                    binF = 0.0f;
                else
                    binF = angle / step;
                int bin = static_cast<int> (std::floor(binF));

                CV_Assert(bin >= 0 && bin < nBins);
                float binDist = abs(binF - bin - 0.5f);

                float weightA = weight * (1.0f - binDist);
                float weightB = weight * binDist;
                hist[bin] += weightA;

                if (2 * (binF - bin) < step)
                    hist[(bin + nBins - 1) % nBins] += weightB;
                else
                    hist[(bin + 1) % nBins] += weightB;

                deltaAngle += step;
            }

            int bestBin = -1;
            float maxBin = -1;
            for (int i = 0; i < nBins; i++)
            {
                if (hist[i] > maxBin)
                {
                    maxBin = hist[i];
                    bestBin = i;
                }
            }

            int l = (bestBin == 0) ? nBins - 1 : bestBin - 1;
            int r = (bestBin + 1) % nBins;
            float bestAngle2 = bestBin + 0.5f * ((hist[l]) - (hist[r])) / ((hist[l]) - 2.0f * (hist[bestBin]) + (hist[r]));
            bestAngle2 = (bestAngle2 < 0) ? nBins + bestAngle2 : (bestAngle2 >= nBins) ? bestAngle2 - nBins : bestAngle2;
            bestAngle2 *= step;

            return bestAngle2;
        }

        void MSDDetector_Impl::nonMaximaSuppression(std::vector< std::vector<float> > & saliency, std::vector<cv::KeyPoint> & keypoints)
        {
            cv::KeyPoint kp_temp;
            int border = m_search_area_radius + m_patch_radius;

            std::vector<cv::Point2f> orientPoints;
            if (m_compute_orientation)
            {
                int nBins = 36;
                float step = float((2 * CV_PI) / nBins);
                float deltaAngle = 0.0f;

                for (int i = 0; i < nBins; i++)
                {
                    cv::Point2f pt;
                    pt.x = m_search_area_radius * cos(deltaAngle);
                    pt.y = m_search_area_radius * sin(deltaAngle);

                    orientPoints.push_back(pt);

                    deltaAngle += step;
                }
            }

            for (int r = 0; r < m_cur_n_scales; r++)
            {
                int cW = m_scaleSpace[r].cols;
                int cH = m_scaleSpace[r].rows;

                for (int j = border; j < cH - border; j++)
                {
                    for (int i = border; i < cW - border; i++)
                    {
                        if (saliency[r][j * cW + i] <= m_th_saliency)
                            continue;

                        if (m_mask.rows > 0)
                        {
                            int j_full = cvRound(j * std::pow(m_scale_factor, r));
                            int i_full = cvRound(i * std::pow(m_scale_factor, r));
                            if ((int) m_mask.at<unsigned char>(j_full, i_full) == 0)
                                continue;
                        }

                        bool is_max = true;

                        for (int k = cv::max(0, r - m_nms_scale_radius); k <= cv::min(m_cur_n_scales - 1, r + m_nms_scale_radius); k++)
                        {
                            if (k != r)
                            {
                                int j_sc = cvRound(j * std::pow(m_scale_factor, r - k));
                                int i_sc = cvRound(i * std::pow(m_scale_factor, r - k));

                                if (saliency[r][j * cW + i] < saliency[k][j_sc * cW + i_sc])
                                {
                                    is_max = false;
                                    break;
                                }
                            }
                        }

                        for (int v = cv::max(border, j - m_nms_radius); v <= cv::min(cH - border - 1, j + m_nms_radius); v++)
                        {
                            for (int u = cv::max(border, i - m_nms_radius); u <= cv::min(cW - border - 1, i + m_nms_radius); u++)
                            {
                                if (saliency[r][j * cW + i] < saliency[r][v * cW + u])
                                {
                                    is_max = false;
                                    break;
                                }
                            }

                            if (!is_max)
                                break;
                        }

                        if (is_max)
                        {
                            bool resInt = rescalePoint(i, j, r, saliency, kp_temp.pt);
                            if (!resInt)
                                continue;


                            if (m_mask.rows > 0)
                            {
                                if (m_mask.at<unsigned char>((int) kp_temp.pt.y, (int) kp_temp.pt.x) == 0)
                                    continue;
                            }
                            kp_temp.response = saliency[r][j * cW + i];
                            kp_temp.size = (m_patch_radius * 2.0f + 1) * std::pow(m_scale_factor, r);
                            kp_temp.octave = r;
                            if (m_compute_orientation)
                                kp_temp.angle = computeOrientation(m_scaleSpace[r], i, j, orientPoints);

                            keypoints.push_back(kp_temp);
                        }
                    }
                }
            }

        }

        Ptr<MSDDetector> MSDDetector::create(int m_patch_radius, int m_search_area_radius,
                int m_nms_radius, int m_nms_scale_radius, float m_th_saliency, int m_kNN, float m_scale_factor,
                int m_n_scales, bool m_compute_orientation)
        {
            return makePtr<MSDDetector_Impl>(m_patch_radius, m_search_area_radius,
                    m_nms_radius, m_nms_scale_radius, m_th_saliency, m_kNN, m_scale_factor,
                    m_n_scales, m_compute_orientation);
        }

    }
}
