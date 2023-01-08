// This implementation of, and any deviation from, the original algorithm as
// proposed by Ziegler et al. is not endorsed by Ziegler et al. nor does it
// claim to represent their definition of locally uniform comparison image
// descriptor. The original LUCID algorithm as proposed by Ziegler et al. remains
// the property of its respective authors. This implementation is an adaptation of
// said algorithm and contributed to OpenCV by Str3iber.

// References:
// Ziegler, Andrew, Eric Christiansen, David Kriegman, and Serge J. Belongie.
// "Locally uniform comparison image descriptor." In Advances in Neural Information
// Processing Systems, pp. 1-9. 2012.

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
*/

#include "precomp.hpp"

namespace cv {
    namespace xfeatures2d {
        /*!
         LUCID implementation
         */
        class LUCIDImpl : public LUCID {
            public:
                /** Constructor
                 * @param lucid_kernel kernel for descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
                 * @param blur_kernel kernel for blurring image prior to descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
                 */
                LUCIDImpl(const int lucid_kernel = 1, const int blur_kernel = 2);

                void read( const FileNode& fn) CV_OVERRIDE;
                void write( FileStorage& fs) const CV_OVERRIDE;

                /**
                 * @param lucid_kernel kernel for descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
                 */
                void setLucidKernel(int lucid_kernel) CV_OVERRIDE { l_kernel = lucid_kernel; }
                /**
                 * @return lucid kernel, where 1=3x3, 2=5x5, 3=7x7 and so forth
                 */
                int getLucidKernel() const CV_OVERRIDE { return l_kernel; }

                /**
                 * @param blur_kernel kernel for blurring image prior to descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
                 */
                void setBlurKernel(int blur_kernel) CV_OVERRIDE { b_kernel = blur_kernel*2+1; }
                /**
                 * @return blur kernel, where 1=3x3, 2=5x5, 3=7x7 and so forth
                 */
                int getBlurKernel() const CV_OVERRIDE { return (b_kernel-1)/2; }

                /** returns the descriptor length */
                virtual int descriptorSize() const CV_OVERRIDE;

                /** returns the descriptor type */
                virtual int descriptorType() const CV_OVERRIDE;

                /** returns the default norm type */
                virtual int defaultNorm() const CV_OVERRIDE;

                virtual void compute(InputArray _src, std::vector<KeyPoint> &keypoints, OutputArray _desc) CV_OVERRIDE;

            protected:
                int l_kernel, b_kernel;
        };

        Ptr<LUCID> LUCID::create(const int lucid_kernel, const int blur_kernel) {
            return makePtr<LUCIDImpl>(lucid_kernel, blur_kernel);
        }

        LUCIDImpl::LUCIDImpl(const int lucid_kernel, const int blur_kernel) {
            l_kernel = lucid_kernel;
            b_kernel = blur_kernel*2+1;
        }

        void LUCIDImpl::read( const FileNode& fn)
        {
          // if node is empty, keep previous value
          if (!fn["lucid_kernel"].empty())
              fn["lucid_kernel"] >> l_kernel;
          if (!fn["blur_kernel"].empty())
          {
              int _b_kernel;
              fn["blur_kernel"] >> _b_kernel;
              b_kernel = _b_kernel*2+1;
          }
        }
        void LUCIDImpl::write( FileStorage& fs) const
        {
            if(fs.isOpened())
            {
                fs << "name" << getDefaultName();
                fs << "lucid_kernel" << l_kernel;
                fs << "blur_kernel" << (b_kernel-1)/2;
            }
        }

        int LUCIDImpl::descriptorSize() const {
            return (l_kernel*2+1)*(l_kernel*2+1)*3;
        }

        int LUCIDImpl::descriptorType() const {
            return CV_8UC1;
        }

        int LUCIDImpl::defaultNorm() const {
            return NORM_HAMMING;
        }

        // gliese581h suggested filling a cv::Mat with descriptors to enable BFmatcher compatibility
        // speed-ups and enhancements by gliese581h
        void LUCIDImpl::compute(InputArray _src, std::vector<KeyPoint> &keypoints, OutputArray _desc) {
            if (_src.empty()) return;
            CV_Assert(_src.depth() == CV_8U);
            cv::Mat src_input;
            if (_src.channels() == 4)
                cvtColor(_src, src_input, COLOR_BGRA2BGR);
            else {
                CV_Assert(_src.channels() == 3);
                src_input = _src.getMat();
            }

            Mat_<Vec3b> src;

            blur(src_input, src, cv::Size(b_kernel, b_kernel));

            int x, y, j, d, p, m = (l_kernel*2+1)*(l_kernel*2+1)*3, width = src.cols, height = src.rows, r, c;

            Mat_<uchar> desc(static_cast<int>(keypoints.size()), m);

            for (std::size_t i = 0; i < keypoints.size(); ++i) {
                x = static_cast<int>(keypoints[i].pt.x)-l_kernel, y = static_cast<int>(keypoints[i].pt.y)-l_kernel, d = x+2*l_kernel, p = y+2*l_kernel, j = x, r = static_cast<int>(i), c = 0;

                while (x <= d) {
                    Vec3b &pix = src((y < 0 ? height+y : y >= height ? y-height : y), (x < 0 ? width+x : x >= width ? x-width : x));

                    desc(r, c++) = pix[0];
                    desc(r, c++) = pix[1];
                    desc(r, c++) = pix[2];

                    ++x;
                    if (x > d) {
                        if (y < p) {
                            ++y;
                            x = j;
                        }
                        else
                            break;
                    }
                }
            }

            if (_desc.needed())
                sort(desc, _desc, SORT_EVERY_ROW | SORT_ASCENDING);
        }

        String LUCID::getDefaultName() const
        {
          return (Feature2D::getDefaultName() + ".LUCID");
        }
    }
} // END NAMESPACE CV
