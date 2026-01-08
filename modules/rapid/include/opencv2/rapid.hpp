// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_RAPID_HPP_
#define OPENCV_RAPID_HPP_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/**
@defgroup rapid silhouette based 3D object tracking

implements "RAPID-a video rate object tracker" @cite harris1990rapid with the dynamic control point extraction of @cite drummond2002real
*/

namespace cv
{
namespace rapid
{
//! @addtogroup rapid
//! @{

/**
 * Debug draw markers of matched correspondences onto a lineBundle
 * @param bundle the lineBundle
 * @param cols column coordinates in the line bundle
 * @param colors colors for the markers. Defaults to white.
 */
CV_EXPORTS_W void drawCorrespondencies(InputOutputArray bundle, InputArray cols,
                                       InputArray colors = noArray());

/**
 * Debug draw search lines onto an image
 * @param img the output image
 * @param locations the source locations of a line bundle
 * @param color the line color
 */
CV_EXPORTS_W void drawSearchLines(InputOutputArray img, InputArray locations, const Scalar& color);

/**
 * Draw a wireframe of a triangle mesh
 * @param img the output image
 * @param pts2d the 2d points obtained by @ref projectPoints
 * @param tris triangle face connectivity
 * @param color line color
 * @param type line type. See @ref LineTypes.
 * @param cullBackface enable back-face culling based on CCW order
 */
CV_EXPORTS_W void drawWireframe(InputOutputArray img, InputArray pts2d, InputArray tris,
                                const Scalar& color, int type = LINE_8, bool cullBackface = false);
/**
 * Extract control points from the projected silhouette of a mesh
 *
 * see @cite drummond2002real Sec 2.1, Step b
 * @param num number of control points
 * @param len search radius (used to restrict the ROI)
 * @param pts3d the 3D points of the mesh
 * @param rvec rotation between mesh and camera
 * @param tvec translation between mesh and camera
 * @param K camera intrinsic
 * @param imsize size of the video frame
 * @param tris triangle face connectivity
 * @param ctl2d the 2D locations of the control points
 * @param ctl3d matching 3D points of the mesh
 */
CV_EXPORTS_W void extractControlPoints(int num, int len, InputArray pts3d, InputArray rvec, InputArray tvec,
                                       InputArray K, const Size& imsize, InputArray tris, OutputArray ctl2d,
                                       OutputArray ctl3d);
/**
 * Extract the line bundle from an image
 * @param len the search radius. The bundle will have `2*len + 1` columns.
 * @param ctl2d the search lines will be centered at this points and orthogonal to the contour defined by
 * them. The bundle will have as many rows.
 * @param img the image to read the pixel intensities values from
 * @param bundle line bundle image with size `ctl2d.rows() x (2 * len + 1)` and the same type as @p img
 * @param srcLocations the source pixel locations of @p bundle in @p img as CV_16SC2
 */
CV_EXPORTS_W void extractLineBundle(int len, InputArray ctl2d, InputArray img, OutputArray bundle,
                                    OutputArray srcLocations);

/**
 * Find corresponding image locations by searching for a maximal sobel edge along the search line (a single
 * row in the bundle)
 * @param bundle the line bundle
 * @param cols correspondence-position per line in line-bundle-space
 * @param response the sobel response for the selected point
 */
CV_EXPORTS_W void findCorrespondencies(InputArray bundle, OutputArray cols,
                                       OutputArray response = noArray());

/**
 * Collect corresponding 2d and 3d points based on correspondencies and mask
 * @param cols correspondence-position per line in line-bundle-space
 * @param srcLocations the source image location
 * @param pts2d 2d points
 * @param pts3d 3d points
 * @param mask mask containing non-zero values for the elements to be retained
 */
CV_EXPORTS_W void convertCorrespondencies(InputArray cols, InputArray srcLocations, OutputArray pts2d,
                                          InputOutputArray pts3d = noArray(), InputArray mask = noArray());

/**
 * High level function to execute a single rapid @cite harris1990rapid iteration
 *
 * 1. @ref extractControlPoints
 * 2. @ref extractLineBundle
 * 3. @ref findCorrespondencies
 * 4. @ref convertCorrespondencies
 * 5. @ref solvePnPRefineLM
 *
 * @param img the video frame
 * @param num number of search lines
 * @param len search line radius
 * @param pts3d the 3D points of the mesh
 * @param tris triangle face connectivity
 * @param K camera matrix
 * @param rvec rotation between mesh and camera. Input values are used as an initial solution.
 * @param tvec translation between mesh and camera. Input values are used as an initial solution.
 * @param rmsd the 2d reprojection difference
 * @return ratio of search lines that could be extracted and matched
 */
CV_EXPORTS_W float rapid(InputArray img, int num, int len, InputArray pts3d, InputArray tris, InputArray K,
                         InputOutputArray rvec, InputOutputArray tvec, CV_OUT double* rmsd = 0);

/// Abstract base class for stateful silhouette trackers
class CV_EXPORTS_W Tracker : public Algorithm
{
public:
    virtual ~Tracker();
    CV_WRAP virtual float
    compute(InputArray img, int num, int len, InputArray K, InputOutputArray rvec, InputOutputArray tvec,
            const TermCriteria& termcrit = TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 5, 1.5)) = 0;
    CV_WRAP virtual void clearState() = 0;
};

/// wrapper around @ref rapid function for uniform access
class CV_EXPORTS_W Rapid : public Tracker
{
public:
    CV_WRAP static Ptr<Rapid> create(InputArray pts3d, InputArray tris);
};

/** implements "Optimal local searching for fast and robust textureless 3D object tracking in highly
 * cluttered backgrounds" @cite seo2013optimal
 */
class CV_EXPORTS_W OLSTracker : public Tracker
{
public:
    CV_WRAP static Ptr<OLSTracker> create(InputArray pts3d, InputArray tris, int histBins = 8, uchar sobelThesh = 10);
};

/** implements "Global optimal searching for textureless 3D object tracking" @cite wang2015global
 */
class CV_EXPORTS_W GOSTracker : public Tracker
{
public:
    CV_WRAP static Ptr<OLSTracker> create(InputArray pts3d, InputArray tris, int histBins = 4, uchar sobelThesh = 10);
};
//! @}
} /* namespace rapid */
} /* namespace cv */

#endif /* OPENCV_RAPID_HPP_ */
