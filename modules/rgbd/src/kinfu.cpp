// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"

namespace cv {
namespace kinfu {


// MatType should be Mat or UMat
template< typename MatType>
class KinFuImpl : public KinFu
{
public:
    KinFuImpl();
    virtual ~KinFuImpl();

    void render(OutputArray image) const CV_OVERRIDE;
    void render(OutputArray image, const Matx44f& cameraPose) const CV_OVERRIDE;

    virtual void getCloud(OutputArray points, OutputArray normals) const CV_OVERRIDE;
    void getPoints(OutputArray points) const CV_OVERRIDE;
    void getNormals(InputArray points, OutputArray normals) const CV_OVERRIDE;

    void reset() CV_OVERRIDE;

    const Affine3f getPose() const CV_OVERRIDE;

    bool update(InputArray depth) CV_OVERRIDE;

    bool updateT(const MatType& depth);

private:
    OdometrySettings odometrySettings;
    Odometry icp;

    VolumeSettings volumeSettings;
    Volume volume;

    int frameCounter;
    Matx44f pose;
    OdometryFrame prevFrame;
    OdometryFrame renderFrame;

    float tsdf_min_camera_movement = 0.f; //meters, disabled
    Vec3f lightPose = Vec3f::all(0.f);
};


template< typename MatType >
KinFuImpl<MatType>::KinFuImpl()
{
    volumeSettings = VolumeSettings(VolumeType::TSDF);
    volume = Volume(VolumeType::TSDF, volumeSettings);

    Matx33f intr;
    volumeSettings.getCameraIntegrateIntrinsics(intr);
    const Vec4i volumeDims;
    volumeSettings.getVolumeDimentions(volumeDims);
    const float voxelSize = volumeSettings.getVoxelSize();

    odometrySettings = OdometrySettings();
    odometrySettings.setMaxRotation(30.f);
    odometrySettings.setMaxTranslation(voxelSize * (float)volumeDims[0] * 0.5f);
    odometrySettings.setCameraMatrix(intr);
    icp = Odometry(OdometryType::DEPTH, odometrySettings, OdometryAlgoType::FAST);

    reset();
}

template< typename MatType >
void KinFuImpl<MatType >::reset()
{
    frameCounter = 0;
    pose = Affine3f::Identity().matrix;
    volume.reset();
}

template< typename MatType >
KinFuImpl<MatType>::~KinFuImpl()
{ }

template< typename MatType >
const Affine3f KinFuImpl<MatType>::getPose() const
{
    return pose;
}


template<>
bool KinFuImpl<Mat>::update(InputArray _depth)
{
    Size frameSize(volumeSettings.getWidth(), volumeSettings.getHeight());
    CV_Assert(!_depth.empty() && _depth.size() == frameSize);

    Mat depth;
    if(_depth.isUMat())
    {
        _depth.copyTo(depth);
        return updateT(depth);
    }
    else
    {
        return updateT(_depth.getMat());
    }
}


template<>
bool KinFuImpl<UMat>::update(InputArray _depth)
{
    Size frameSize(volumeSettings.getWidth(), volumeSettings.getHeight());
    CV_Assert(!_depth.empty() && _depth.size() == frameSize);

    UMat depth;
    if(!_depth.isUMat())
    {
        _depth.copyTo(depth);
        return updateT(depth);
    }
    else
    {
        return updateT(_depth.getUMat());
    }
}


template< typename MatType >
bool KinFuImpl<MatType>::updateT(const MatType& _depth)
{
    CV_TRACE_FUNCTION();

    MatType depth;
    if(_depth.type() != DEPTH_TYPE)
        _depth.convertTo(depth, DEPTH_TYPE);
    else
        depth = _depth;

    OdometryFrame newFrame = icp.createOdometryFrame();
    newFrame.setDepth(depth);

    Size frameSize(volumeSettings.getWidth(), volumeSettings.getHeight());

    if(frameCounter == 0)
    {
        icp.prepareFrame(newFrame);
        // use depth instead of distance
        volume.integrate(depth, pose);
    }
    else
    {
        Affine3f affine;
        Matx44d mrt;
        Mat Rt;
        icp.prepareFrames(newFrame, prevFrame);
        bool success = icp.compute(newFrame, prevFrame, Rt);
        if(!success)
            return false;
        affine.matrix = Matx44f(Rt);
        pose = (Affine3f(pose) * affine).matrix;

        float rnorm = (float)cv::norm(affine.rvec());
        float tnorm = (float)cv::norm(affine.translation());
        // We do not integrate volume if camera does not move
        if((rnorm + tnorm)/2 >= tsdf_min_camera_movement)
        {
            // use depth instead of distance
            volume.integrate(depth, pose);
        }

        MatType points, normals;
        newFrame.getPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
        newFrame.getPyramidAt(normals, OdometryFramePyramidType::PYR_NORM,  0);
        volume.raycast(pose, frameSize.height, frameSize.width, points, normals);

        newFrame.setPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
        newFrame.setPyramidAt(normals, OdometryFramePyramidType::PYR_NORM,  0);
    }

    renderFrame = newFrame;
    prevFrame = newFrame;

    frameCounter++;
    return true;
}


template< typename MatType >
void KinFuImpl<MatType>::render(OutputArray image) const
{
    CV_TRACE_FUNCTION();
    MatType pts, nrm;
    renderFrame.getPyramidAt(pts, OdometryFramePyramidType::PYR_CLOUD, 0);
    renderFrame.getPyramidAt(nrm, OdometryFramePyramidType::PYR_NORM,  0);
    detail::renderPointsNormals(pts, nrm, image, lightPose);
}


template< typename MatType >
void KinFuImpl<MatType>::render(OutputArray image, const Matx44f& _cameraPose) const
{
    CV_TRACE_FUNCTION();

    Affine3f cameraPose(_cameraPose);
    MatType points, normals;
    Size frameSize(volumeSettings.getWidth(), volumeSettings.getHeight());
    volume.raycast(_cameraPose, frameSize.height, frameSize.width, points, normals);
    detail::renderPointsNormals(points, normals, image, lightPose);
}


template< typename MatType >
void KinFuImpl<MatType>::getCloud(OutputArray p, OutputArray n) const
{
    volume.fetchPointsNormals(p, n);
}


template< typename MatType >
void KinFuImpl<MatType>::getPoints(OutputArray points) const
{
    volume.fetchPointsNormals(points, noArray());
}


template< typename MatType >
void KinFuImpl<MatType>::getNormals(InputArray points, OutputArray normals) const
{
    volume.fetchNormals(points, normals);
}

// importing class

#ifdef OPENCV_ENABLE_NONFREE

Ptr<KinFu> KinFu::create()
{
#ifdef HAVE_OPENCL
    if(cv::ocl::useOpenCL())
        return makePtr< KinFuImpl<UMat> >();
#endif
        return makePtr< KinFuImpl<Mat> >();
}

#else
Ptr<KinFu> KinFu::create()
{
    CV_Error(Error::StsNotImplemented,
             "This algorithm is patented and is excluded in this configuration; "
             "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}
#endif

KinFu::~KinFu() {}

} // namespace kinfu
} // namespace cv
