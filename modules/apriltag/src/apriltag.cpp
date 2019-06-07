// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/apriltag.hpp"

#include <apriltag.h>
#include <apriltag_pose.h>
#include <tag16h5.h>
#include <tag25h9.h>
#include <tag36h11.h>
#include <tagCircle21h7.h>
#include <tagCircle49h12.h>
#include <tagCustom48h12.h>
#include <tagStandard41h12.h>
#include <tagStandard52h13.h>

//TODO:
#include <iostream>

namespace cv {
namespace apriltag {

using namespace std;

static void copyVector2Output(vector<vector<Point2d> >& vec, OutputArrayOfArrays out)
{
    const int outputDepth = out.depth();
    out.create(static_cast<int>(vec.size()), 1, CV_MAKETYPE(outputDepth, 2));

    if (out.isMatVector())
    {
        for (int i = 0; i < static_cast<int>(vec.size()); i++)
        {
            out.create(4, 1, CV_MAKETYPE(outputDepth, 2), i);
            Mat& m = out.getMatRef(i);
            Mat(Mat(vec[i]).t()).convertTo(m, outputDepth);
        }
    }
    else if (out.isUMatVector())
    {
        for (int i = 0; i < static_cast<int>(vec.size()); i++)
        {
            out.create(4, 1, CV_MAKETYPE(outputDepth, 2), i);
            UMat& m = out.getUMatRef(i);
            Mat(Mat(vec[i]).t()).convertTo(m, outputDepth);
        }
    }
    else if (out.kind() == _OutputArray::STD_VECTOR_VECTOR)
    {
        for (int i = 0; i < static_cast<int>(vec.size()); i++)
        {
            out.create(4, 1, CV_MAKETYPE(outputDepth, 2), i);
            Mat m = out.getMat(i);
            Mat(Mat(vec[i]).t()).convertTo(m, outputDepth);
        }
    }
    else
    {
        CV_Error(cv::Error::StsNotImplemented,
                 "Only Mat vector, UMat vector, and vector<vector> OutputArrays are currently supported.");
    }
}

static void getTagObjectPoints(double tagSize, vector<Point3d>& objectPoints)
{
    objectPoints.resize(4);

    // define coordinates system
    const double tagSize_2 = tagSize / 2;
    objectPoints[0] = Point3d(-tagSize_2,  tagSize_2, 0);
    objectPoints[1] = Point3d( tagSize_2,  tagSize_2, 0);
    objectPoints[2] = Point3d( tagSize_2, -tagSize_2, 0);
    objectPoints[3] = Point3d(-tagSize_2, -tagSize_2, 0);
}

static void convertPoseToMat(const apriltag_pose_t& pose, Matx31d& rvec, Matx31d& tvec)
{
    Matx33d R;

    for (unsigned int i = 0; i < 3; i++)
    {
        for (unsigned int j = 0; j < 3; j++)
        {
            R(i,j) = MATD_EL(pose.R, i, j);
        }
        tvec(i,0) = MATD_EL(pose.t, i, 0);
    }

    Rodrigues(R, rvec);
}

static void drawDetectedTags(InputOutputArray _img, InputArrayOfArrays _corners, InputArray _ids,
                             const Scalar& borderColor, int thickness, double fontSize)
{
    Mat img = _img.getMat();
    CV_Check(img.empty(), !img.empty(), "Image must not be empty");
    const int type = img.type();
    CV_CheckType(type, type == CV_8UC1 || type == CV_8UC3, "Image must be CV_8UC1 or CV_8UC3 type");
    const int nTags = static_cast<int>(_corners.total());
    CV_Assert((_corners.total() == _ids.total()) || _ids.empty());
    if (!_ids.empty())
    {
        Mat matIds = _ids.getMat();
        CV_CheckDepth(matIds.depth(), CV_32S, "Vector of tag ids must be stored in int type");
    }

    // calculate colors
    Scalar textColor, cornerColor;
    textColor = cornerColor = borderColor;
    swap(textColor.val[0], textColor.val[1]);     // text color just sawp G and R
    swap(cornerColor.val[1], cornerColor.val[2]); // corner color just sawp G and B

    for (int i = 0; i < nTags; i++)
    {
        Mat currentMarker_ = _corners.getMat(i), currentMarker;
        if (currentMarker_.depth() != CV_64F)
        {
            currentMarker_.convertTo(currentMarker, CV_64F);
        }
        else
        {
            currentMarker = currentMarker_;
        }
        CV_CheckEQ(static_cast<int>(currentMarker.total()), 4, "Number of tag corners must be 4");

        // draw marker sides
        for (int j = 0; j < 4; j++)
        {
            Point2d p0 = currentMarker.ptr<Point2d>(0)[j];
            Point2d p1 = currentMarker.ptr<Point2d>(0)[(j + 1) % 4];
            line(img, p0, p1, borderColor, thickness);
        }

        // draw first corner mark
        Point2d ptLength = (currentMarker.ptr<Point2d>(0)[2] - currentMarker.ptr<Point2d>(0)[0]) / 20;
        double length = (std::max)((std::max)(std::fabs(ptLength.x), std::fabs(ptLength.y)), 3.0);
        Point2d rectBegin = currentMarker.ptr<Point2d>(0)[0] - Point2d(length, length);
        Point2d rectEnd = currentMarker.ptr<Point2d>(0)[0] + Point2d(length, length);
        rectangle(img, rectBegin, rectEnd, cornerColor, thickness, LINE_AA);

        // draw ID
        if (_ids.total() > 0)
        {
            Point2d cent(0, 0);
            for (int p = 0; p < 4; p++)
            {
                cent += currentMarker.ptr<Point2d>(0)[p];
            }
            cent = cent / 4.;
            stringstream s;
            putText(img, format("id: %d", _ids.getMat().ptr<int>(0)[i]), cent, FONT_HERSHEY_SIMPLEX,
                    fontSize, textColor, thickness);
        }
    }
}

static void estimateTagPosePnP(InputArray _objPoints, InputArray _corners, InputArray _cameraMatrix,
                               InputArray _distCoeffs, pair<Matx31d, Matx31d>& _rvecs,
                               pair<Matx31d, Matx31d>& _tvecs, SolvePnPMethod pnpMethod,
                               bool useExtrinsicGuess, vector<pair<double, double> >* _reprojectionError)
{
    vector<Matx31d> rvecs, tvecs;
    vector<double> reprojError;
    if (solvePnPGeneric(_objPoints, _corners, _cameraMatrix, _distCoeffs, rvecs, tvecs,
                        useExtrinsicGuess, pnpMethod, _rvecs.first, _tvecs.first,
                        _reprojectionError ? reprojError : noArray()))
    {
        if (rvecs.size() == 2)
        {
            _rvecs = pair<Matx31d, Matx31d>(rvecs[0], rvecs[1]);
            _tvecs = pair<Matx31d, Matx31d>(tvecs[0], tvecs[1]);

            if (_reprojectionError)
            {
                _reprojectionError->push_back(pair<double, double>(reprojError[0], reprojError[1]));
            }
        }
        else
        {
            _rvecs = pair<Matx31d, Matx31d>(rvecs[0], rvecs[0]);
            _tvecs = pair<Matx31d, Matx31d>(tvecs[0], tvecs[0]);

            if (_reprojectionError)
            {
                _reprojectionError->push_back(pair<double, double>(reprojError[0], reprojError[0]));
            }
        }
    }
}

struct AprilTagDetector::Impl
{
    Impl(const AprilTagFamily& tagFamily) :
        m_detections(NULL), m_tagFamily(tagFamily), m_td(NULL), m_tf(NULL)
    {
        switch(m_tagFamily)
        {
        case TAG_16h5:
            m_tf = tag16h5_create();
            break;

        case TAG_25h9:
            m_tf = tag25h9_create();
            break;

        case TAG_36h11:
            m_tf = tag36h11_create();
            break;

        case TAG_CIRCLE21h7:
            m_tf = tagCircle21h7_create();
            break;

        case TAG_CIRCLE49h12:
            m_tf = tagCircle49h12_create();
            break;

        case TAG_CUSTOM48h12:
            m_tf = tagCustom48h12_create();
            break;

        case TAG_STANDARD41h12:
            m_tf = tagStandard41h12_create();
            break;

        case TAG_STANDARD52h13:
            m_tf = tagStandard52h13_create();
            break;

        default:
            break;
        }

        m_td = apriltag_detector_create();
        apriltag_detector_add_family(m_td, m_tf);
    }

    ~Impl()
    {
        apriltag_detector_destroy(m_td);

        switch(m_tagFamily)
        {
        case TAG_16h5:
            tag16h5_destroy(m_tf);
            break;

        case TAG_25h9:
            tag25h9_destroy(m_tf);
            break;

        case TAG_36h11:
            tag36h11_destroy(m_tf);
            break;

        case TAG_CIRCLE21h7:
            tagCircle21h7_destroy(m_tf);
            break;

        case TAG_CIRCLE49h12:
            tagCircle49h12_destroy(m_tf);
            break;

        case TAG_CUSTOM48h12:
            tagCustom48h12_destroy(m_tf);
            break;

        case TAG_STANDARD41h12:
            tagStandard41h12_destroy(m_tf);
            break;

        case TAG_STANDARD52h13:
            tagStandard52h13_destroy(m_tf);
            break;

        default:
            break;
        }

        if (m_detections)
        {
            apriltag_detections_destroy(m_detections);
        }
    }

    void detectTags(const Mat& img, vector<vector<Point2d> >& corners, vector<int>& ids)
    {
        image_u8_t im = {/*.width =*/ static_cast<int32_t>(img.cols),
                         /*.height =*/ static_cast<int32_t>(img.rows),
                         /*.stride =*/ static_cast<int32_t>(img.cols),
                         /*.buf =*/ img.data};

        if (m_detections)
        {
            apriltag_detections_destroy(m_detections);
            m_detections = NULL;
        }

        m_detections = apriltag_detector_detect(m_td, &im);

        int nbDetections = zarray_size(m_detections);
        corners.resize(static_cast<size_t>(nbDetections));
        ids.resize(static_cast<size_t>(nbDetections));

        for (int i = 0; i < nbDetections; i++)
        {
            apriltag_detection_t *det;
            zarray_get(m_detections, i, &det);

            vector<Point2d> tagCorners(4);
            for (size_t j = 0; j < tagCorners.size(); j++)
            {
                tagCorners[j].x = det->p[j][0];
                tagCorners[j].y = det->p[j][1];
            }
            corners[i] = tagCorners;
            ids[i] = det->id;
        }
    }

    void drawTag(InputOutputArray image, const Size& size, int id)
    {
        image_u8_t *im = apriltag_to_image(m_tf, id);
        Mat imgTag = Mat::zeros(im->height, im->width, CV_8UC1);

        //TODO: hotfix
        if (string(m_tf->name) == "tag16h5" ||
            string(m_tf->name) == "tag25h9" ||
            string(m_tf->name) == "tag36h11")
        {
            //create white border
            for (int i = 0; i < imgTag.rows; i++)
            {
                imgTag.at<uchar>(i,0) = 255;
                imgTag.at<uchar>(i,imgTag.cols-1) = 255;
            }
            for (int j = 0; j < imgTag.cols; j++)
            {
                imgTag.at<uchar>(0,j) = 255;
                imgTag.at<uchar>(imgTag.rows-1,j) = 255;
            }

            //Copy inner tag
            for (int i = 2; i < imgTag.rows-2; i++)
            {
                for (int j = 2; j < imgTag.cols-2; j++)
                {
                    imgTag.at<uchar>(i,j) = im->buf[i*im->stride + j];
                }
            }
        }
        else
        {
            for (int i = 0; i < imgTag.rows; i++)
            {
                for (int j = 0; j < imgTag.cols; j++)
                {
                    imgTag.at<uchar>(i,j) = im->buf[i*im->stride + j];
                }
            }
        }

        image_u8_destroy(im);

        image.create(size, CV_8UC1);
        Mat ref = image.getMat();
        resize(imgTag, ref, size, 0, 0, INTER_NEAREST);
    }

    void estimateTagsPoseAprilTag(double _tagSize, InputArray _cameraMatrix,
                                  InputArray _distCoeffs, vector<pair<Matx31d, Matx31d> >& _rvecs,
                                  vector<pair<Matx31d, Matx31d> >& _tvecs,
                                  vector<pair<double, double> >* _reprojectionError,
                                  OutputArray _objPoints)
    {
        CV_Check(_tagSize, _tagSize > 0, "Tag length must be > 0");

        int nbDetections = zarray_size(m_detections);
        if (nbDetections <= 0)
        {
            _rvecs.clear();
            _tvecs.clear();

            if (_reprojectionError)
            {
                _reprojectionError->clear();
            }

            return;
        }

        _rvecs.resize(nbDetections);
        _tvecs.resize(nbDetections);
        if (_reprojectionError)
        {
            _reprojectionError->resize(nbDetections);
        }

        Mat cameraMatrix0 = _cameraMatrix.getMat();
        Mat distCoeffs0 = _distCoeffs.getMat();
        Mat cameraMatrix = Mat_<double>(cameraMatrix0);
        Mat distCoeffs = Mat_<double>(distCoeffs0);
        double fx = cameraMatrix.at<double>(0,0), fy = cameraMatrix.at<double>(1,1);
        double cx = cameraMatrix.at<double>(0,2), cy = cameraMatrix.at<double>(1,2);

        // for each marker, calculate its pose
        for (int i = 0; i < nbDetections; i++)
        {
            apriltag_detection_t *det;
            zarray_get(m_detections, i, &det);

            apriltag_detection_info_t info;
            info.det = det;
            info.tagsize = _tagSize;
            info.fx = fx;
            info.fy = fy;
            info.cx = cx;
            info.cy = cy;

            apriltag_pose_t pose1, pose2;
            double err_1 = (numeric_limits<double>::max)(), err_2 = (numeric_limits<double>::max)();
            estimate_tag_pose_orthogonal_iteration(&info, &err_1, &pose1, &err_2, &pose2, 50);

            Matx31d rvec1, tvec1;
            Matx31d rvec2, tvec2;
            if (err_1 <= err_2)
            {
                convertPoseToMat(pose1, rvec1, tvec1);

                if (pose2.R)
                {
                    convertPoseToMat(pose2, rvec2, tvec2);
                }
                else
                {
                    rvec2 = rvec1;
                    tvec2 = tvec1;
                }

                if (_reprojectionError)
                {
                    (*_reprojectionError)[i] = pair<double, double>(err_1, err_2);
                }
            }
            else
            {
                convertPoseToMat(pose2, rvec1, tvec1);
                convertPoseToMat(pose1, rvec2, tvec2);

                if (_reprojectionError)
                {
                    (*_reprojectionError)[i] = pair<double, double>(err_2, err_1);
                }
            }

            matd_destroy(pose1.R);
            matd_destroy(pose1.t);
            if (pose2.R)
            {
                matd_destroy(pose2.t);
            }
            matd_destroy(pose2.R);

            _rvecs[i].first = rvec1;
            _tvecs[i].first = tvec1;

            _rvecs[i].second = rvec2;
            _tvecs[i].second = tvec2;
        }

        if (_objPoints.needed())
        {
            vector<Point3d> tagObjPoints;
            getTagObjectPoints(_tagSize, tagObjPoints);

            Mat(tagObjPoints).convertTo(_objPoints, _objPoints.depth());
        }
    }

    void setDecodeSharpening(double decodeSharpening)
    {
        m_td->decode_sharpening = decodeSharpening;
    }

    void setNumThreads(int nThreads)
    {
        m_td->nthreads = nThreads;
    }

    void setQuadDecimate(float quadDecimate)
    {
        m_td->quad_decimate = quadDecimate;
    }

    void setQuadSigma(float quadSigma)
    {
        m_td->quad_sigma = quadSigma;
    }

    void setRefineEdges(bool refineEdges)
    {
        m_td->refine_edges = refineEdges ? 1 : 0;
    }

    zarray_t *m_detections;
    AprilTagFamily m_tagFamily;
    apriltag_detector_t *m_td;
    apriltag_family_t *m_tf;
};

AprilTagDetector::AprilTagDetector(const AprilTagFamily& tagFamily) : pImpl(new Impl(tagFamily))
{
}

AprilTagDetector::~AprilTagDetector()
{
    delete pImpl;
}

void AprilTagDetector::detectTags(InputArray _img, OutputArrayOfArrays _corners, OutputArray _ids)
{
    Mat img0 = _img.getMat();
    CV_Check(img0.empty(), !img0.empty(), "Image must not be empty");
    const int type = img0.type();
    CV_CheckType(type, type == CV_8UC1 || type == CV_8UC3 || type == CV_8UC4,
                 "Image must be CV_8UC1, CV_8UC3 or CV_8UC4 type");

    Mat img;
    if (img0.type() == CV_8UC1)
    {
        img = img0;
    }
    else if (img0.type() == CV_8UC3)
    {
        cvtColor(img0, img, COLOR_BGR2GRAY);
    }
    else if (img0.type() == CV_8UC4)
    {
        cvtColor(img0, img, COLOR_BGRA2GRAY);
    }

    vector<vector<Point2d> > corners;
    vector<int> ids;
    pImpl->detectTags(img, corners, ids);

    copyVector2Output(corners, _corners);
    Mat(ids).copyTo(_ids);
}

void AprilTagDetector::drawDetectedTags(InputOutputArray img, InputArrayOfArrays corners, InputArray ids,
                                        const Scalar& borderColor, int thickness, double fontSize)
{
    cv::apriltag::drawDetectedTags(img, corners, ids, borderColor, thickness, fontSize);
}

void AprilTagDetector::estimateTagsPoseAprilTag(double tagSize, InputArray cameraMatrix, InputArray distCoeffs,
                                                vector<pair<Matx31d, Matx31d> >& rvecs, vector<pair<Matx31d, Matx31d> >& tvecs,
                                                std::vector<std::pair<double, double> >* reprojectionError, OutputArray objPoints)
{
    pImpl->estimateTagsPoseAprilTag(tagSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojectionError, objPoints);
}

void AprilTagDetector::estimateTagsPosePnP(InputArrayOfArrays corners, double tagSize, InputArray cameraMatrix,
                                           InputArray distCoeffs, vector<pair<Matx31d, Matx31d> >& rvecs,
                                           vector<pair<Matx31d, Matx31d> >& tvecs, SolvePnPMethod pnpMethod,
                                           const vector<bool>& useExtrinsicGuesses,
                                           std::vector<std::pair<double, double> >* reprojectionError,
                                           OutputArray objectPoints)
{
    if (objectPoints.needed())
    {
        vector<vector<Point3d> > objPts;
        estimateTagsPosePnP(corners, vector<double>(corners.total(), tagSize), cameraMatrix, distCoeffs, rvecs, tvecs,
                            vector<SolvePnPMethod>(corners.total(), pnpMethod), useExtrinsicGuesses, reprojectionError, objPts);

        if (!objPts.empty())
        {
            Mat(objPts.front()).convertTo(objectPoints, objectPoints.depth());
        }
    }
    else
    {
        estimateTagsPosePnP(corners, vector<double>(corners.total(), tagSize), cameraMatrix, distCoeffs, rvecs, tvecs,
                            vector<SolvePnPMethod>(corners.total(), pnpMethod), useExtrinsicGuesses, reprojectionError);
    }
}

void AprilTagDetector::estimateTagsPosePnP(InputArrayOfArrays corners, const vector<double>& tagsSize, InputArray cameraMatrix,
                                           InputArray distCoeffs, vector<pair<Matx31d, Matx31d> >& rvecs,
                                           vector<pair<Matx31d, Matx31d> >& tvecs, const vector<SolvePnPMethod>& pnpMethods,
                                           const vector<bool>& useExtrinsicGuesses,
                                           std::vector<std::pair<double, double> >* reprojectionError,
                                           OutputArrayOfArrays objectsPoints)
{
    size_t nTags = corners.total();
    if (nTags == 0)
    {
        rvecs.clear();
        tvecs.clear();

        if (reprojectionError)
        {
            reprojectionError->clear();
        }

        return;
    }

    size_t nTagsSize = tagsSize.size();
    size_t nPnPMethods = pnpMethods.size();
    size_t nGuesses = useExtrinsicGuesses.size();
    CV_CheckEQ(nTags, nTagsSize, "Number of tags and number of tags size must match");
    CV_CheckEQ(nTags, nPnPMethods, "Number of tags and number of PnP methods must match");
    if (nGuesses > 0)
    {
        CV_CheckEQ(nTags, nGuesses, "Number of tags and number of extrinsic guess flags must match");
        CV_CheckEQ(nTags, rvecs.size(), "Number of tags and number of initial rotation vectors must match");
        CV_CheckEQ(nTags, tvecs.size(), "Number of tags and number of initial translation vectors must match");
    }
    else
    {
        rvecs.resize(nTags);
        tvecs.resize(nTags);
    }

    bool returnObjectPoints = objectsPoints.needed();
    int objectPointsDepth = -1;
    if (returnObjectPoints)
    {
        CV_CheckEQ(objectsPoints.kind(), _InputArray::STD_VECTOR_VECTOR,
                   "objectsPoints must be vector<vector<Point3f/Point3d> > type");
        objectPointsDepth = objectsPoints.depth();
        CV_CheckDepth(objectPointsDepth, objectPointsDepth == CV_32F || objectPointsDepth == CV_64F,
                      "objectsPoints must be Point3f or Point3d");
        int channels = objectsPoints.channels();
        CV_CheckChannelsEQ(channels, 3, "objectsPoints must be Point3f or Point3d");

        objectsPoints.create(static_cast<int>(nTags), 1, CV_MAKETYPE(objectPointsDepth, 3));
    }

    if (reprojectionError)
    {
        reprojectionError->clear();
    }

    for (size_t i = 0; i < nTags; i++)
    {
        double tagSize = tagsSize[i];
        CV_CheckGT(tagSize, 0.0, "Tag size must be > 0");

        vector<Point3d> tagObjPoints;
        getTagObjectPoints(tagSize, tagObjPoints);

        bool useExtrinsicGuess = useExtrinsicGuesses.empty() ? false : useExtrinsicGuesses[i];
        SolvePnPMethod tag_pnp = pnpMethods[i];
        estimateTagPosePnP(tagObjPoints, corners.getMat(static_cast<int>(i)), cameraMatrix,
                           distCoeffs, rvecs[i], tvecs[i], tag_pnp, useExtrinsicGuess, reprojectionError);

        if (returnObjectPoints)
        {
            objectsPoints.create(1, static_cast<int>(tagObjPoints.size()), CV_MAKETYPE(objectPointsDepth, 3), static_cast<int>(i));
            Mat ref = objectsPoints.getMat_(static_cast<int>(i));

            if (objectPointsDepth == CV_32F)
            {
                for (size_t j = 0; j < tagObjPoints.size(); j++)
                {
                    ref.at<Vec3f>(0,static_cast<int>(j)) = Vec3f(static_cast<float>(tagObjPoints[j].x),
                                                                 static_cast<float>(tagObjPoints[j].y),
                                                                 static_cast<float>(tagObjPoints[j].z));
                }
            }
            else if (objectPointsDepth == CV_64F)
            {
                for (size_t j = 0; j < tagObjPoints.size(); j++)
                {
                    ref.at<Vec3d>(0,static_cast<int>(j)) = tagObjPoints[j];
                }
            }
        }
    }
}

void AprilTagDetector::drawTag(InputOutputArray image, const Size& size, int id)
{
    pImpl->drawTag(image, size, id);
}

void AprilTagDetector::setDecodeSharpening(double decodeSharpening)
{
    pImpl->setDecodeSharpening(decodeSharpening);
}

void AprilTagDetector::setNumThreads(int nThreads)
{
    pImpl->setNumThreads(nThreads);
}

void AprilTagDetector::setQuadDecimate(float quadDecimate)
{
    pImpl->setQuadDecimate(quadDecimate);
}

void AprilTagDetector::setQuadSigma(float quadSigma)
{
    pImpl->setQuadSigma(quadSigma);
}

void AprilTagDetector::setRefineEdges(bool refineEdges)
{
    pImpl->setRefineEdges(refineEdges);
}

}
}
