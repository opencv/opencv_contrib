/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.
                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.
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
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef __OPENCV_XIMGPROC_SEGMENTATION_HPP__
#define __OPENCV_XIMGPROC_SEGMENTATION_HPP__

#include <opencv2/core.hpp>

namespace cv {
    namespace ximgproc {
        namespace segmentation {
            //! @addtogroup ximgproc_segmentation
            //! @{

                    /** @brief Graph Based Segmentation Algorithm.
                        The class implements the algorithm described in @cite PFF2004 .
                     */
                    class CV_EXPORTS_W GraphSegmentation : public Algorithm {
                        public:
                            /** @brief Segment an image and store output in dst
                                @param src The input image. Any number of channel (1 (Eg: Gray), 3 (Eg: RGB), 4 (Eg: RGB-D)) can be provided
                                @param dst The output segmentation. It's a CV_32SC1 Mat with the same number of cols and rows as input image, with an unique, sequential, id for each pixel.
                            */
                            CV_WRAP virtual void processImage(InputArray src, OutputArray dst) = 0;

                            CV_WRAP virtual void setSigma(double sigma) = 0;
                            CV_WRAP virtual double getSigma() = 0;

                            CV_WRAP virtual void setK(float k) = 0;
                            CV_WRAP virtual float getK() = 0;

                            CV_WRAP virtual void setMinSize(int min_size) = 0;
                            CV_WRAP virtual int getMinSize() = 0;
                    };

                    /** @brief Creates a graph based segmentor
                        @param sigma The sigma parameter, used to smooth image
                        @param k The k parameter of the algorythm
                        @param min_size The minimum size of segments
                     */
                    CV_EXPORTS_W Ptr<GraphSegmentation> createGraphSegmentation(double sigma=0.5, float k=300, int min_size=100);
            //! @}

            // Represent an edge between two pixels
            class Edge {
                public:
                    int from;
                    int to;
                    float weight;

                    bool operator <(const Edge& e) const {
                        return weight < e.weight;
                    }
            };

            // A point in the sets of points
            class PointSetElement {
                public:
                    int p;
                    int size;

                    PointSetElement() { }

                    PointSetElement(int p_) {
                        p = p_;
                        size = 1;
                    }
            };

            // An object to manage set of points, who can be fusionned
            class PointSet {
                public:
                    PointSet(int nb_elements_);
                    ~PointSet();

                    int nb_elements;

                    // Return the main point of the point's set
                    int getBasePoint(int p);

                    // Join two sets of points, based on their main point
                    void joinPoints(int p_a, int p_b);

                    // Return the set size of a set (based on the main point)
                    int size(unsigned int p) { return mapping[p].size; }

                private:
                    PointSetElement* mapping;

            };

        }
    }
}

#endif
