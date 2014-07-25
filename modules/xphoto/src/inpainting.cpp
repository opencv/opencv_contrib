/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <time.h>

#include "opencv2/xphoto.hpp"
#include "../../../../opencv_main/modules/imgproc/src/gcgraph.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/stitching.hpp"

#include "opencv2/core.hpp"
#include "opencv2/core/core_c.h"

#include "opencv2/core/types.hpp"
#include "opencv2/core/types_c.h"

namespace cv
{
    template <typename Tp1, typename Tp2> static inline Tp2 sqr(Tp1 arg) { return arg * arg; }

    template <typename Tp> static inline Tp sqr(Tp arg) { return arg * arg; }

    static inline float norm2(const float &a, const float &b) { return sqr(a - b); }

    static inline float norm2(const Vec2f &a, const Vec2f &b) { return (a - b).dot(a - b); }

    static inline float norm2(const Vec3f &a, const Vec3f &b) { return (a - b).dot(a - b); }

    static inline float norm2(const Vec4f &a, const Vec4f &b) { return (a - b).dot(a - b); }

    #define dist(imgs, l1, l2, idx1, idx2) ( norm2(imgs[l1](idx1), imgs[l2](idx1)) \
                                           + norm2(imgs[l1](idx2), imgs[l2](idx2)) )

    template <typename Tp>
    static inline void setWeights(GCGraph <float> &graph,
        const std::vector < Mat_<Tp> > &imgs, const int A, const int B,
        const int labelA, const int labelB, const int alpha,
        const Point &pointA, const Point &pointB)
    {
        //************************************************************//
        //************************************************************//

        if (labelA == labelB)
        {
            double weightAB = dist( imgs, labelA, alpha, pointA, pointB );
            graph.addEdges(A, B, weightAB, weightAB);
        }
        else
        {
            double weightAX = dist( imgs, labelA, alpha, pointA, pointB );
            double weightXB = dist( imgs, alpha, labelB, pointA, pointB );
            double weightXSink = dist( imgs, labelA, labelB, pointA, pointB );

            int X = graph.addVtx();

            graph.addEdges(A, X, weightAX, weightAX);
            graph.addEdges(X, B, weightXB, weightXB);
            graph.addTermWeights(X, 0, weightXSink);
        }
    }

    template <typename Tp>
    static double alphaExpansion(const std::vector < Mat_<Tp> > &imgs,
        const std::vector < Mat_<uchar> > &masks,
        const Mat_<int> &labeling, const int alpha, Mat_<int> &nlabeling)
    {
        //************************************************************//
        //************************************************************//

        const int height = imgs[0].rows;
        const int width = imgs[0].cols;

        const double infinity = 10000000000;

        const int actualEdges = height*(width - 1) + width*(height - 1);
        GCGraph <float> graph(height*width + actualEdges, 2*actualEdges);

        // terminal links
        for (int i = 0; i < height; ++i)
        {
            const uchar *maskAlphaRow = masks[alpha].ptr(i);
            const int *labelRow = (const int *) labeling.ptr(i);

            for (int j = 0; j < width; ++j)
                graph.addTermWeights( graph.addVtx(),
                                      maskAlphaRow[j] ? 0 : infinity,
                           masks[ labelRow[j] ](i, j) ? 0 : infinity );
        }

        // neighbor links
        for (int i = 0; i < height - 1; ++i)
        {
            const int *currentRow = (const int *) labeling.ptr(i);
            const int *nextRow = (const int *) labeling.ptr(i + 1);

            for (int j = 0; j < width - 1; ++j)
            {
                setWeights( graph, imgs, i*width + j,   i*width + (j + 1),
                                         currentRow[j], currentRow[j + 1], alpha,
                                         Point(i, j),     Point(i, j + 1) );
                setWeights( graph, imgs, i*width + j,   (i + 1)*width + j,
                                         currentRow[j],        nextRow[j], alpha,
                                         Point(i, j),      Point(i + 1, j) );
            }
        }

        double result = graph.maxFlow();

        nlabeling.create( labeling.size() );
        for (int i = 0; i < height; ++i)
        {
            const int *inRow = (const int *) labeling.ptr(i);
            int *outRow = (int *) nlabeling.ptr(i);

            for (int j = 0; j < width; ++j)
            {
                bool gPart = graph.inSourceSegment(i*width + j);
                outRow[j] = gPart ? inRow[j] : alpha;
            }
        }

        return result;
    }

    template <typename Tp>
    static void shiftMapInpaint(const Mat_<Tp> &src, const Mat_<uchar> &mask, Mat_<Tp> &dst)
    {
        //************************************************************//
        //************************************************************//

        const int nTransform = 60; // number of dominant transforms for stitching
        const int psize = 8; // single ANNF patch size

        const int width = src.cols;
        const int height = src.rows;

        Mat_<uchar> invMask = 255 - mask;
        dilate(invMask, invMask, Mat(), Point(-1,-1), 2);

        /** Downsample **/
        //...

        /** ANNF computation **/
        int init = time(NULL);
        srand( 1406297336 );
        std::cout << init << std::endl;

        std::vector <Matx33f> transforms; // dominant transforms
        for (int i = 0; i < nTransform; ++i)
        {
            float dx   = rand()%width - width/2;
            float dy = rand()%height - height/2;
            transforms.push_back( Matx33f( 1, 0, dx,
                                           0, 1, dy,
                                           0, 0,  1) );
        }

        /** Warping **/
        std::vector < Mat_<Tp> > imgs( nTransform + 1 );      // source image transformed with transforms[i]
        std::vector < Mat_<uchar> > masks( nTransform + 1 );  // validity mask for current shift

        src.copyTo( imgs[0] );
        mask.copyTo( masks[0] );

        for (int i = 0; i < nTransform; ++i)
        {
            Mat_<Tp> nsrc( src.size() );
            warpPerspective( src, nsrc, transforms[i], src.size(),
                             INTER_LINEAR, BORDER_CONSTANT, 0 );

            Mat_<uchar> nmask( mask.size(), mask.type() );
            warpPerspective( mask, nmask, transforms[i], mask.size(),
                             INTER_NEAREST, BORDER_CONSTANT, 0 );
            nmask &= invMask;

            nsrc.copyTo( imgs[i + 1] );
            nmask.copyTo( masks[i + 1] );
        }

        /** Stitching **/
        std::vector <double> costs( nTransform + 1 );
        std::vector < Mat_<int> > labelings( nTransform + 1 );

        Mat_<int> labeling( height, width, 0 );
        double cost = std::numeric_limits<double>::max();

        for (int success = false, num = 0; ; success = false)
        {
            for (int i = 0; i < nTransform + 1; ++i)
                costs[i] = alphaExpansion(imgs, masks, labeling, i, labelings[i]);

            for (int i = 0; i < nTransform + 1; ++i)
                if (costs[i] < 0.98*cost)
                {
                    success = true;
                    cost = costs[num = i];
                }

            if (success == false)
                break;

            labelings[num].copyTo(labeling);
        }

        for (int k = 0; k < height*width; ++k)
        {
            int i = k / width;
            int j = k % width;
            dst(i, j) = imgs[labeling(i, j)](i, j);
        }

        /** Upsample and refinement **/
        //...
    }

    template <typename Tp>
    void inpaint(const Mat_<Tp> src, const Mat_<uchar> mask, Mat_<Tp> dst, const int algorithmType)
    {
        //************************************************************//
        //************************************************************//

        dst.create( src.size() );

        switch ( algorithmType )
        {
            case INPAINT_SHIFTMAP:
                shiftMapInpaint(src, mask, dst);
                break;
            default:
                CV_Assert( false );
                break;
        }
    }

    /*! The function reconstructs the selected image area from known area.
    *  \param src : source image.
    *  \param mask : inpainting mask, 8-bit 1-channel image. Zero pixels indicate the area that needs to be inpainted.
    *  \param dst : destination image.
    *  \param algorithmType : inpainting method.
    */
    void inpaint(const Mat &src, const Mat &mask, Mat &dst, const int algorithmType)
    {
        //************************************************************//
        //************************************************************//

        CV_Assert( mask.channels() == 1 && mask.depth() == CV_8U );
        CV_Assert( src.rows == mask.rows && src.cols == mask.cols );

        switch ( src.type() )
        {
            case CV_8UC1:
                inpaint( Mat_<uchar>(src), Mat_<uchar>(mask), Mat_<uchar>(dst), algorithmType );
                break;
            case CV_8UC2:
                inpaint( Mat_<Vec2b>(src), Mat_<uchar>(mask), Mat_<Vec2b>(dst), algorithmType );
                break;
            case CV_8UC3:
                inpaint( Mat_<Vec3b>(src), Mat_<uchar>(mask), Mat_<Vec3b>(dst), algorithmType );
                break;
            case CV_8UC4:
                inpaint( Mat_<Vec4b>(src), Mat_<uchar>(mask), Mat_<Vec4b>(dst), algorithmType );
                break;
            case CV_16SC1:
                inpaint( Mat_<short>(src), Mat_<uchar>(mask), Mat_<short>(dst), algorithmType );
                break;
            case CV_16SC2:
                inpaint( Mat_<Vec2s>(src), Mat_<uchar>(mask), Mat_<Vec2s>(dst), algorithmType );
                break;
            case CV_16SC3:
                inpaint( Mat_<Vec3s>(src), Mat_<uchar>(mask), Mat_<Vec3s>(dst), algorithmType );
                break;
            case CV_16SC4:
                inpaint( Mat_<Vec4s>(src), Mat_<uchar>(mask), Mat_<Vec4s>(dst), algorithmType );
                break;
            case CV_32SC1:
                inpaint( Mat_<int>(src), Mat_<uchar>(mask), Mat_<int>(dst), algorithmType );
                break;
            case CV_32SC2:
                inpaint( Mat_<Vec2i>(src), Mat_<uchar>(mask), Mat_<Vec2i>(dst), algorithmType );
                break;
            case CV_32SC3:
                inpaint( Mat_<Vec3i>(src), Mat_<uchar>(mask), Mat_<Vec3i>(dst), algorithmType );
                break;
            case CV_32SC4:
                inpaint( Mat_<Vec4i>(src), Mat_<uchar>(mask), Mat_<Vec4i>(dst), algorithmType );
                break;
            case CV_32FC1:
                inpaint( Mat_<float>(src), Mat_<uchar>(mask), Mat_<float>(dst), algorithmType);
                break;
            case CV_32FC2:
                inpaint( Mat_<Vec2f>(src), Mat_<uchar>(mask), Mat_<Vec2f>(dst), algorithmType );
                break;
            case CV_32FC3:
                inpaint( Mat_<Vec3f>(src), Mat_<uchar>(mask), Mat_<Vec3f>(dst), algorithmType );
                break;
            case CV_32FC4:
                inpaint( Mat_<Vec4f>(src), Mat_<uchar>(mask), Mat_<Vec4f>(dst), algorithmType );
                break;
            case CV_64FC1:
                inpaint( Mat_<double>(src), Mat_<uchar>(mask), Mat_<double>(dst), algorithmType );
                break;
            case CV_64FC2:
                inpaint( Mat_<Vec2d>(src), Mat_<uchar>(mask), Mat_<Vec2d>(dst), algorithmType );
                break;
            case CV_64FC3:
                inpaint( Mat_<Vec3d>(src), Mat_<uchar>(mask), Mat_<Vec3d>(dst), algorithmType );
                break;
            case CV_64FC4:
                inpaint( Mat_<Vec4d>(src), Mat_<uchar>(mask), Mat_<Vec4d>(dst), algorithmType );
                break;
            default:
                CV_Assert( false );
                break;
        }
    }
}