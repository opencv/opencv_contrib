// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

/*
 * Functions to perform affine adaptation of keypoint and to calculate descriptors of elliptic regions
 */

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"

namespace {

using namespace cv;
using namespace cv::xfeatures2d;

/*
* Functions to perform affine adaptation of circular keypoint
*/
void calcAffineCovariantRegions(const Mat& image, const std::vector<KeyPoint>& keypoints, std::vector<Elliptic_KeyPoint>& affRegions);
void calcAffineCovariantDescriptors( const Ptr<DescriptorExtractor>& dextractor, const Mat& img, std::vector<Elliptic_KeyPoint>& affRegions, Mat& descriptors );

void calcSecondMomentMatrix(const Mat & dx2, const Mat & dxy, const Mat & dy2, Point p, Matx22f& M);
bool calcAffineAdaptation(const Mat & image, Elliptic_KeyPoint& keypoint);
float selIntegrationScale(const Mat & image, float si, Point c);
float selDifferentiationScale(const Mat & image, Mat & Lxm2smooth, Mat & Lxmysmooth, Mat & Lym2smooth, float si, Point c);
float calcSecondMomentSqrt(const Mat & dx2, const Mat & dxy, const Mat & dy2, Point p, Matx22f& Mk);
float normMaxEval(Matx22f & U, Mat& uVal, Mat& uVect);

/*
 * Calculates second moments matrix in point p
 */
void calcSecondMomentMatrix(const Mat & dx2, const Mat & dxy, const Mat & dy2, Point p, Matx22f & M)
{
    int x = p.x;
    int y = p.y;

    M(0, 0) = dx2.at<float> (y, x);
    M(0, 1) = M(1, 0) = dxy.at<float> (y, x);
    M(1, 1) = dy2.at<float> (y, x);
}

/*
 * Performs affine adaptation
 */
bool calcAffineAdaptation(const Mat & fimage, Elliptic_KeyPoint & keypoint)
{
    Matx23f transf; /*Transformation matrix*/
    Matx21f   size; /*Image size after transformation*/
    Matx21f      c; /*Transformed point*/
    Matx21f      p; /*Image point*/

    Matx22f U(1.f, 0.f, 0.f, 1.f); /*Normalization matrix*/

    Mat warpedImg, Lxm2smooth, Lym2smooth, Lxmysmooth, img_roi;
    Matx22f Mk;
    float Qinv = 1, q, si = keypoint.si;
    bool divergence = false, convergence = false;
    int i = 0;

    //Coordinates in image
    int py = (int) keypoint.pt.y;
    int px = (int) keypoint.pt.x;

    //Roi coordinates
    int roix, roiy;

    //Coordinates in U-trasformation
    int cx = px;
    int cy = py;
    int cxPr = cx;
    int cyPr = cy;

    float radius = keypoint.size / 2 * 1.4f;
    float half_width, half_height;

    Rect roi;
    float ax1, ax2;
    float phi = 0;
    ax1 = ax2 = keypoint.size / 2;
    Mat drawImg;

    //Affine adaptation
    while (i <= 10 && !divergence && !convergence)
    {
        //Transformation matrix
        transf = Matx23f(
            U(0,0), U(0,1), 0.f,
            U(1,0), U(1,1), 0.f
        );
        keypoint.transf = transf;

        Size_<float> boundingBox;

        float ac_b2 = float(determinant(U));
        boundingBox.width  = ceil(U(1, 1)/ac_b2 * 3 * si*1.4f );
        boundingBox.height = ceil(U(0, 0)/ac_b2 * 3 * si*1.4f );

        //Create window around interest point
        half_width = std::min((float) std::min(fimage.cols - px-1, px), boundingBox.width);
        half_height = std::min((float) std::min(fimage.rows - py-1, py), boundingBox.height);
        roix = max(px - (int) boundingBox.width, 0);
        roiy = max(py - (int) boundingBox.height, 0);
        roi = Rect(roix, roiy, px - roix + int(half_width)+1, py - roiy + int(half_height)+1);

        //create ROI
        img_roi = fimage(roi);


        //Point within the ROI
        p(0, 0) = float(px - roix);
        p(1, 0) = float(py - roiy);

        if (half_width <= 0 || half_height <= 0)
            return divergence;

        //Find coordinates of square's angles to find size of warped ellipse's bounding box
        float u00 = U(0, 0);
        float u01 = U(0, 1);
        float u10 = U(1, 0);
        float u11 = U(1, 1);

        float minx = u01 * img_roi.rows < 0 ? u01 * img_roi.rows : 0;
        float miny = u10 * img_roi.cols < 0 ? u10 * img_roi.cols : 0;
        float maxx = (u00 * img_roi.cols > u00 * img_roi.cols + u01 * img_roi.rows ? u00
                * img_roi.cols : u00 * img_roi.cols + u01 * img_roi.rows) - minx;
        float maxy = (u11 * img_roi.rows > u10 * img_roi.cols + u11 * img_roi.rows ? u11
                * img_roi.rows : u10 * img_roi.cols + u11 * img_roi.rows) - miny;

        //Shift
        transf(0, 2) = -minx;
        transf(1, 2) = -miny;

        /*float min_width = minx >= 0 ? u00 * img_roi.cols - u01 * img_roi.rows : u00 * img_roi.cols
                + u01 * img_roi.rows;
        float min_height = miny >= 0 ? u11 * img_roi.rows - u10 * img_roi.cols : u10 * img_roi.cols
                + u11 * img_roi.rows;*/

        if (maxx >=  2*radius+1 && maxy >=  2*radius+1)
        {
            //Size of normalized window must be 2*radius
            //Transformation
            Mat warpedImgRoi;
            warpAffine(img_roi, warpedImgRoi, transf, Size(int(maxx), int(maxy)),INTER_AREA, BORDER_REPLICATE);

            //Point in U-Normalized coordinates
            c = U * p;
            cx = int(c(0, 0) - minx);
            cy = int(c(1, 0) - miny);

            if (warpedImgRoi.rows > 2 * radius+1 && warpedImgRoi.cols > 2 * radius+1)
            {
                //Cut around normalized patch
                roix = std::max(cx - int(ceil(radius)), 0);
                roiy = std::max(cy - int(ceil(radius)), 0);
                roi = Rect(roix, roiy,
                        cx - roix + std::min(int(ceil(radius)), warpedImgRoi.cols - cx-1)+1,
                        cy - roiy + std::min(int(ceil(radius)), warpedImgRoi.rows - cy-1)+1);
                warpedImg = warpedImgRoi(roi);

                //Coordinates in cutted ROI
                cx = cx - roix;
                cy = cy - roiy;
            } else
                warpedImgRoi.copyTo(warpedImg);

            //Integration Scale selection
            si = selIntegrationScale(warpedImg, si, Point(cx, cy));
            //Differentation scale selection
            selDifferentiationScale(warpedImg, Lxm2smooth, Lxmysmooth, Lym2smooth, si,
                    Point(cx, cy));
            if (Lym2smooth.empty()) {
                divergence = true;
                continue;
            }

            //Spatial Localization
            cxPr = cx; //Previous iteration point in normalized window
            cyPr = cy;

            float cornMax = 0;
            for (int j = 0; j < 3; j++)
            {
                for (int t = 0; t < 3; t++)
                {
                    float dx2 = Lxm2smooth.at<float> (cyPr - 1 + j, cxPr - 1 + t);
                    float dy2 = Lym2smooth.at<float> (cyPr - 1 + j, cxPr - 1 + t);
                    float dxy = Lxmysmooth.at<float> (cyPr - 1 + j, cxPr - 1 + t);
                    float det = dx2 * dy2 - dxy * dxy;
                    float tr = dx2 + dy2;
                    float cornerness = det - (0.04f * tr*tr);
                    if (cornerness > cornMax)
                    {
                        cornMax = cornerness;
                        cx = cxPr - 1 + t;
                        cy = cyPr - 1 + j;
                    }
                }
            }

            //Transform point in image coordinates
            p(0, 0) = float(px);
            p(1, 0) = float(py);
            //Displacement vector
            c(0, 0) = float(cx - cxPr);
            c(1, 0) = float(cy - cyPr);
            //New interest point location in image
            p = p + Matx22f(Matx22d(U).inv()) * c;
            px = int(p(0, 0));
            py = int(p(1, 0));

            q = calcSecondMomentSqrt(Lxm2smooth, Lxmysmooth, Lym2smooth, Point(cx, cy), Mk);

            float ratio = 1 - q;

            //if ratio == 1 means q == 0 and one axes equals to 0
            if (!cvIsNaN(ratio) && ratio != 1)
            {
                //Update U matrix
                U = U * Mk;

                Mat uVal, uV;
                eigen(U, uVal, uV);

                Qinv = normMaxEval(U, uVal, uV);

                //Keypoint doesn't converge
                if (Qinv >= 6)
                    divergence = true;

                //Keypoint converges
                else if (ratio <= 0.05f)
                {
                    convergence = true;

                    //Set transformation matrix
                    transf = Matx23f(
                        U(0,0), U(0,1), 0.f,
                        U(1,0), U(1,1), 0.f
                    );
                    keypoint.transf = transf;

                    ax1 = 1.f / std::abs(uVal.at<float> (0, 0)) * 3 * si;
                    ax2 = 1.f / std::abs(uVal.at<float> (1, 0)) * 3 * si;
                    phi = float(atan(uV.at<float> (1, 0) / uV.at<float> (0, 0)) * (180) / CV_PI);
                    keypoint.axes = Size_<float> (ax1, ax2);
                    keypoint.angle = phi;
                    keypoint.pt = Point2f( (float) px, (float) py);
                    keypoint.si = si;
                    keypoint.size = 2 * 3 * si;

                } else
                    radius = 3 * si * 1.4f;

            } else divergence = true;

        } else divergence = true;

        ++i;
    }

    return convergence;
}

/*
 * Selects the integration scale that maximize LoG in point c
 */
float selIntegrationScale(const Mat & image, float si, Point c)
{
    Mat Lap, L;
    int cx = c.x;
    int cy = c.y;
    float maxLap = 0;
    float maxsx = si;
    int gsize;
    float sigma, sigma_prev = 0;

    image.copyTo(L);
    /* Search best integration scale between previous and successive layer
     */
    for (float u = 0.7f; u <= 1.41f; u += 0.1f)
    {
        float sik = u * si;
        sigma = sqrt(powf(sik, 2) - powf(sigma_prev, 2));

        gsize = int(ceil(sigma * 3)) * 2 + 1;

        GaussianBlur(L, L, Size(gsize, gsize), sigma);
        sigma_prev = sik;

        Laplacian(L, Lap, CV_32F, 3);

        float lapVal = sik * sik * std::abs(Lap.at<float> (cy, cx));

        if (u == 0.7f)
            maxLap = lapVal;

        if (lapVal >= maxLap)
        {
            maxLap = lapVal;
            maxsx = sik;
        }
    }
    return maxsx;
}

/*
 * Calculates second moments matrix square root
 */
float calcSecondMomentSqrt(const Mat & dx2, const Mat & dxy, const Mat & dy2, Point p, Matx22f & Mk)
{
    Mat V, eigVal, Vinv, D;
    Matx22f M;

    calcSecondMomentMatrix(dx2, dxy, dy2, p, M);

    /* *
     * M = V * D * V.inv()
     * V has eigenvectors as columns
     * D is a diagonal Matrix with eigenvalues as elements
     * V.inv() is the inverse of V
     * */

    eigen(M, eigVal, V);
    V = V.t();
    Vinv = V.inv();

    float eval1 = eigVal.at<float> (0, 0) = sqrt(eigVal.at<float> (0, 0));
    float eval2 = eigVal.at<float> (1, 0) = sqrt(eigVal.at<float> (1, 0));

    D = Mat::diag(eigVal);

    //square root of M
    Mk = Mat(V * D * Vinv);
    //return q isotropic measure
    return min(eval1, eval2) / max(eval1, eval2);
}

float normMaxEval(Matx22f & U, Mat & uVal, Mat & uVec)
{
    /* *
     * Decomposition:
     * U = V * D * V.inv()
     * V has eigenvectors as columns
     * D is a diagonal Matrix with eigenvalues as elements
     * V.inv() is the inverse of V
     * */
    uVec = uVec.t();
    Mat uVinv = uVec.inv();

    //Normalize min eigenvalue to 1 to expand patch in the direction of min eigenvalue of U.inv()
    float uval1 = uVal.at<float> (0, 0);
    float uval2 = uVal.at<float> (1, 0);

    if (std::abs(uval1) < std::abs(uval2))
    {
        uVal.at<float> (0, 0) = 1;
        uVal.at<float> (1, 0) = uval2 / uval1;
    } else
    {
        uVal.at<float> (1, 0) = 1;
        uVal.at<float> (0, 0) = uval1 / uval2;
    }

    Mat D = Mat::diag(uVal);
    //U normalized
    U = Mat(uVec * D * uVinv);

    return max(std::abs(uVal.at<float> (0, 0)), std::abs(uVal.at<float> (1, 0))) / min(
            std::abs(uVal.at<float> (0, 0)), std::abs(uVal.at<float> (1, 0))); //define the direction of warping
}

/*
 * Selects diffrentiation scale
 */
float selDifferentiationScale(const Mat & img, Mat & Lxm2smooth, Mat & Lxmysmooth,
        Mat & Lym2smooth, float si, Point c)
{
    float s = 0.5f;
    float sdk = s * si;
    float sigma_prev = 0, sigma;

    Mat L, dx2, dxy, dy2;

    double qMax = 0;

    //Gaussian kernel size
    int gsize;
    Size ksize;

    img.copyTo(L);

    while (s <= 0.751f)
    {
        Matx22f M;
        float sd = s * si;

        //Smooth previous smoothed image L
        sigma = sqrt(powf(sd, 2) - powf(sigma_prev, 2));

        gsize = int(ceil(sigma * 3)) * 2 + 1;

        GaussianBlur(L, L, Size(gsize, gsize), sigma);

        sigma_prev = sd;

        //X and Y derivatives
        Mat Lx, Ly;
        Sobel(L, Lx, L.depth(), 1, 0, 1);
        Lx = Lx * sd;
        Sobel(L, Ly, L.depth(), 0, 1, 1);
        Ly = Ly * sd;

        //Size of gaussian kernel
        gsize = int(ceil(si * 3)) * 2 + 1;
        ksize = Size(gsize, gsize);

        Mat Lxm2 = Lx.mul(Lx);
        GaussianBlur(Lxm2, dx2, ksize, si);

        Mat Lym2 = Ly.mul(Ly);
        GaussianBlur(Lym2, dy2, ksize, si);

        Mat Lxmy = Lx.mul(Ly);
        GaussianBlur(Lxmy, dxy, ksize, si);

        calcSecondMomentMatrix(dx2, dxy, dy2, Point(c.x, c.y), M);

        //calc eigenvalues
        Mat eval;
        eigen(M, eval);
        double eval1 = std::abs(eval.at<float> (0, 0));
        double eval2 = std::abs(eval.at<float> (1, 0));
        double m = max(eval1, eval2);
        double q = (m == 0) ? -1 : min(eval1, eval2) / m;

        if (q >= qMax)
        {
            qMax = q;
            sdk = sd;
            dx2.copyTo(Lxm2smooth);
            dxy.copyTo(Lxmysmooth);
            dy2.copyTo(Lym2smooth);

        }
        s += 0.05f;
    }

    return sdk;
}

void calcAffineCovariantRegions(const Mat & image, const std::vector<KeyPoint> & keypoints,
        std::vector<Elliptic_KeyPoint> & affRegions)
{
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        KeyPoint kp = keypoints[i];
        Elliptic_KeyPoint ex(kp.pt, 0, Size_<float> (kp.size / 2, kp.size / 2), kp.size,
                kp.size / 6);

        if (calcAffineAdaptation(image, ex))
            affRegions.push_back(ex);
    }
    //Erase similar keypoint
    float maxDiff = 4;
    Mat colorimg;
    for (size_t i = 0; i < affRegions.size(); i++)
    {
        Elliptic_KeyPoint kp1 = affRegions[i];
        for (size_t j = i+1; j < affRegions.size(); j++){

            Elliptic_KeyPoint kp2 = affRegions[j];

            if(norm(kp1.pt-kp2.pt)<=maxDiff){
                float phi1, phi2;
                Size axes1, axes2;
                float si1, si2;
                phi1 = kp1.angle;
                phi2 = kp2.angle;
                axes1 = kp1.axes;
                axes2 = kp2.axes;
                si1 = kp1.si;
                si2 = kp2.si;
                if(std::abs(phi1-phi2)<15 && std::max(si1,si2)/std::min(si1,si2)<1.4f && axes1.width-axes2.width<5 && axes1.height-axes2.height<5){
                    affRegions.erase(affRegions.begin()+j);
                    j--;
                }
            }
        }
    }
}

void calcAffineCovariantDescriptors(const Ptr<DescriptorExtractor>& dextractor, const Mat& img,
        std::vector<Elliptic_KeyPoint>& affRegions, Mat& descriptors)
{

    assert(!affRegions.empty());
    int descriptorSize = dextractor->descriptorSize();
    int descriptorType = dextractor->descriptorType();
    descriptors.create(Size(descriptorSize, int(affRegions.size())), descriptorType);
    descriptors.setTo(0);

    int i = 0;

    for (std::vector<Elliptic_KeyPoint>::iterator it = affRegions.begin(); it < affRegions.end(); ++it)
    {
        Point p = it->pt;

        Matx21f size;
        size(0, 0) = size(1, 0) = it->size;

        //U matrix
        Matx23f transf = it->transf;
        Matx22f U(
            transf(0,0), transf(0,1),
            transf(1,0), transf(1,1)
        );

        float radius = it->size / 2;
        float si = it->si;

        Size_<float> boundingBox;

        float ac_b2 = float(determinant(U));
        boundingBox.width  = ceil(U(1, 1)/ac_b2 * 3 * si );
        boundingBox.height = ceil(U(0, 0)/ac_b2 * 3 * si );

        //Create window around interest point
        float half_width = std::min((float) std::min(img.cols - p.x-1, p.x), boundingBox.width);
        float half_height = std::min((float) std::min(img.rows - p.y-1, p.y), boundingBox.height);
        int roix = max(p.x - (int) boundingBox.width, 0);
        int roiy = max(p.y - (int) boundingBox.height, 0);
        Rect roi = Rect(roix, roiy, p.x - roix + int(half_width)+1, p.y - roiy + int(half_height)+1);

        Mat img_roi = img(roi);

        size(0, 0) = float(img_roi.cols);
        size(1, 0) = float(img_roi.rows);

        size = U * size;

        Mat transfImgRoi, transfImg;
        warpAffine(img_roi, transfImgRoi, transf, Size(int(ceil(size(0, 0))), int(ceil(size(1, 0)))),
                INTER_AREA, BORDER_DEFAULT);

        Matx21f c; //Transformed point
        Matx21f pt; //Image point
        //Point within the Roi
        pt(0, 0) = float(p.x - roix);
        pt(1, 0) = float(p.y - roiy);

        //Point in U-Normalized coordinates
        c = U * pt;
        float cx = c(0, 0);
        float cy = c(1, 0);

        //Cut around point to have patch of 2*keypoint->size

        roix = std::max(int(ceil(cx - radius)), 0);
        roiy = std::max(int(ceil(cy - radius)), 0);

        roi = Rect(roix, roiy, int(ceil(std::min(cx - roix + radius, size(0, 0)))),
                int(ceil(std::min(cy - roiy + radius, size(1, 0)))));
        transfImg = transfImgRoi(roi);

        cx = c(0, 0) - roix;
        cy = c(1, 0) - roiy;

        Mat tmpDesc;
        KeyPoint kp(Point(int(cx), int(cy)), it->size);

        std::vector<KeyPoint> k(1, kp);

        transfImg.convertTo(transfImg, CV_8U);
        dextractor->compute(transfImg, k, tmpDesc);

        tmpDesc.row(0).copyTo(descriptors.row(i));

        i++;

    }

}

} // anonymous namespace

namespace cv
{
namespace xfeatures2d
{
class AffineFeature2D_Impl CV_FINAL : public AffineFeature2D
{
public:
    AffineFeature2D_Impl(
        Ptr<FeatureDetector> keypoint_detector,
        Ptr<DescriptorExtractor> descriptor_extractor
    ) : m_keypoint_detector(keypoint_detector)
      , m_descriptor_extractor(descriptor_extractor) {}
protected:
    using Feature2D::detect; // overload, don't hide
    void detect(InputArray image, std::vector<Elliptic_KeyPoint>& keypoints, InputArray mask) CV_OVERRIDE;
    void detectAndCompute(InputArray image, InputArray mask, std::vector<Elliptic_KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints) CV_OVERRIDE;
    void detectAndCompute(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints) CV_OVERRIDE;
    int descriptorSize() const CV_OVERRIDE;
    int descriptorType() const CV_OVERRIDE;
    int defaultNorm() const CV_OVERRIDE;
private:
    Ptr<FeatureDetector> m_keypoint_detector;
    Ptr<DescriptorExtractor> m_descriptor_extractor;
};

Ptr<AffineFeature2D> AffineFeature2D::create(
    Ptr<FeatureDetector> keypoint_detector,
    Ptr<DescriptorExtractor> descriptor_extractor)
{
    return makePtr<AffineFeature2D_Impl>(keypoint_detector, descriptor_extractor);
}

void AffineFeature2D_Impl::detect(
    InputArray image,
    std::vector<Elliptic_KeyPoint>& keypoints,
    InputArray mask)
{
    std::vector<KeyPoint> non_elliptic_keypoints;
    m_keypoint_detector->detect(image, non_elliptic_keypoints, mask);
    Mat fimage;
    image.getMat().convertTo(fimage, CV_32F, 1.f/255);
    calcAffineCovariantRegions(fimage, non_elliptic_keypoints, keypoints);
}

void AffineFeature2D_Impl::detectAndCompute(
        InputArray image,
        InputArray mask,
        std::vector<Elliptic_KeyPoint>& keypoints,
        OutputArray descriptors,
        bool useProvidedKeypoints)
{
    if(!useProvidedKeypoints)
    {
        std::vector<KeyPoint> non_elliptic_keypoints;
        m_keypoint_detector->detect(image, non_elliptic_keypoints, mask);
        Mat fimage;
        image.getMat().convertTo(fimage, CV_32F, 1.f/255);
        calcAffineCovariantRegions(fimage, non_elliptic_keypoints, keypoints);
    }
    if(descriptors.needed())calcAffineCovariantDescriptors(m_descriptor_extractor, image.getMat(), keypoints, descriptors.getMatRef());
}

void AffineFeature2D_Impl::detectAndCompute(
        InputArray image,
        InputArray mask,
        std::vector<KeyPoint>& keypoints,
        OutputArray descriptors,
        bool useProvidedKeypoints)
{
    if(!useProvidedKeypoints)
    {
        m_keypoint_detector->detect(image, keypoints, mask);
    }
    if(descriptors.needed()) {
        Mat fimage;
        image.getMat().convertTo(fimage, CV_32F, 1.f/255);
        std::vector<Elliptic_KeyPoint> elliptic_keypoints;
        calcAffineCovariantRegions(fimage, keypoints, elliptic_keypoints);
        calcAffineCovariantDescriptors(m_descriptor_extractor, image.getMat(), elliptic_keypoints, descriptors.getMatRef());
    }
}

int AffineFeature2D_Impl::descriptorSize() const
{
    return m_descriptor_extractor->descriptorSize();
}

int AffineFeature2D_Impl::descriptorType() const
{
    return m_descriptor_extractor->descriptorType();
}

int AffineFeature2D_Impl::defaultNorm() const
{
    return m_descriptor_extractor->defaultNorm();
}

}
}
