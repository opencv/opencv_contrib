//
// Created by YIMIN TANG on 10/9/20.
//

#ifndef __OPENCV_PCSEG_H__
#define __OPENCV_PCSEG_H__

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
        CV_EXPORTS
        float angleBetween(const Point3f &, const Point3f &);

        CV_EXPORTS
        bool calCurvatures(
                Mat& ,
                int ,
                std::vector<Point3f>& ,
                std::vector<Point3f>& ,
                std::vector<float>&
        );
        int findFather(std::vector<int>& , int );

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

        CV_EXPORTS
        bool from3dTo2dPlane(
                std::vector<Point3f>& ,
                std::vector<Point3f>& ,
                std::vector<Point2f >&
        );

        CV_EXPORTS
        bool planarMerge(
                std::vector<Point3f>& ,
                std::vector<Point3f>& ,
                int& ,
                std::vector<Point3f>& ,
                std::vector<Point3f>& ,
                float = 0.08
        );

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



#endif //__OPENCV_PCSEG_H__
