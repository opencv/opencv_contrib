// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
//
// Created by YIMIN TANG on 10/9/20.
//

#ifndef __OPENCV_RGBD_PCSEG_HPP__
#define __OPENCV_RGBD_PCSEG_HPP__

#include "opencv2/core/mat.hpp"
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <queue>
#include <algorithm>
#include <cstdio>


namespace cv {
    namespace pcseg {

        /**
         * \brief Calculate the angel between two vectors
         *
         */
        CV_EXPORTS
        float angleBetween(const Point3f &, const Point3f &);

        /**
         * \brief Points Curvature calculation
         *
         */
        CV_EXPORTS
        bool calCurvatures(
                Mat& ,
                int ,
                std::vector<Point3f>& ,
                std::vector<Point3f>& ,
                std::vector<float>&
        );

        /**
         * \brief Used for Disjoint Set, grouping segments
         *
         */
        int findFather(std::vector<int>& , int );

        /**
         * \brief Curvature-Based Plane Segmentation
         *
         */
        CV_EXPORTS
        bool planarSegments(
                std::vector<Point3f>& ,
                std::vector<Point3f>& ,
                std::vector<float>& ,
                int ,
                float ,
                float ,
                std::vector<std::vector<Point3f> >& ,
                std::vector<std::vector<Point3f> >&
        );

        /**
         * \brief Project 3d point to its segmented plane
         *
         */
        bool from3dTo2dPlane(
                std::vector<Point3f>& ,
                std::vector<Point3f>& ,
                std::vector<Point2f >&
        );

        /**
         * \brief Method for merging two planar segments
         *
         */
        CV_EXPORTS
        bool planarMerge(
                std::vector<Point3f>& ,
                std::vector<Point3f>& ,
                int& ,
                std::vector<Point3f>& ,
                std::vector<Point3f>& ,
                float = 0.08
        );

        /**
         * \brief Method for growing planar segments
         *
         */
        CV_EXPORTS
        bool growingPlanar(
                std::vector< std::vector<Point3f> >& ,
                std::vector< std::vector<Point3f> >& ,
                std::vector<int>& ,
                std::vector< std::vector<Point3f> >& ,
                std::vector< std::vector<Point3f> >& ,
                std::vector<int>& ,
                Point3f& ,
                std::vector< std::pair<int,int> >& ,
                float = 15.0/360*2*M_PI,
                int = 7,
                float = 3.75
        );

        /**
         * \brief Merging segments that have grown closer
         *
         */
        CV_EXPORTS
        bool mergeCloseSegments(
                std::vector< std::pair< std::vector<Point3f> ,std::vector<Point3f> > >& ,
                std::vector< std::pair< std::vector<Point3f> ,std::vector<Point3f> > >& ,
                std::vector<int> ,
                std::vector< std::vector<Point3f> >& ,
                std::vector< std::vector<Point3f> >& ,
                std::vector<int>&
        );

    }
}



#endif //__OPENCV_RGBD_PCSEG_HPP__
