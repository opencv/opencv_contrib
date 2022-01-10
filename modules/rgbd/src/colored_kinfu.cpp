// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "opencv2/3d.hpp"

namespace cv {
namespace colored_kinfu {
using namespace kinfu;


// MatType should be Mat or UMat
template< typename MatType>
class ColoredKinFuImpl : public ColoredKinFu
{
public:
    ColoredKinFuImpl();
    virtual ~ColoredKinFuImpl();

    void render(OutputArray image) const CV_OVERRIDE;
    void render(OutputArray image, const Matx44f& cameraPose) const CV_OVERRIDE;

    virtual void getCloud(OutputArray points, OutputArray normals) const CV_OVERRIDE;
    void getPoints(OutputArray points) const CV_OVERRIDE;
    void getNormals(InputArray points, OutputArray normals) const CV_OVERRIDE;

    void reset() CV_OVERRIDE;

    const Affine3f getPose() const CV_OVERRIDE;

    bool update(InputArray depth, InputArray rgb) CV_OVERRIDE;

    bool updateT(const MatType& depth, const MatType& rgb);

private:

    OdometrySettings odometrySettings;
    Odometry icp;

    VolumeSettings volumeSettings;
    Volume volume;

    int frameCounter;
    Matx44f pose;
    OdometryFrame renderFrame;
    OdometryFrame prevFrame;

    std::vector<int> icpIterations = { 5, 3, 2 };
    float tsdf_min_camera_movement = 0.f; //meters, disabled
};


template< typename MatType >
ColoredKinFuImpl<MatType>::ColoredKinFuImpl()
{
    volumeSettings = VolumeSettings(VolumeType::ColorTSDF);
    volume = Volume(VolumeType::ColorTSDF, volumeSettings);

    Matx33f intr;
    volumeSettings.getCameraIntegrateIntrinsics(intr);
    const float voxelSize = volumeSettings.getVoxelSize();
    const Vec4i volumeDims;
    volumeSettings.getVolumeDimentions(volumeDims);

    OdometrySettings ods;
    ods.setCameraMatrix(intr);
    ods.setMaxRotation(30.f);
    ods.setMaxTranslation(voxelSize * (float)volumeDims[0] * 0.5f);
    ods.setIterCounts(icpIterations);

    icp = Odometry(OdometryType::DEPTH, ods, OdometryAlgoType::FAST);

    reset();
}

template< typename MatType >
void ColoredKinFuImpl<MatType >::reset()
{
    frameCounter = 0;
    pose = Affine3f::Identity().matrix;
    volume.reset();
}

template< typename MatType >
ColoredKinFuImpl<MatType>::~ColoredKinFuImpl()
{ }


template< typename MatType >
const Affine3f ColoredKinFuImpl<MatType>::getPose() const
{
    return pose;
}


template<>
bool ColoredKinFuImpl<Mat>::update(InputArray _depth, InputArray _rgb)
{
    Size frameSize(volumeSettings.getWidth(), volumeSettings.getHeight());
    CV_Assert(!_depth.empty() && _depth.size() == frameSize);

    Mat depth;
    Mat rgb;
    if(_depth.isUMat())
    {
        _depth.copyTo(depth);
        _rgb.copyTo(rgb);
        return updateT(depth, rgb);
    }
    else
    {
        return updateT(_depth.getMat(), _rgb.getMat());
    }
}


template<>
bool ColoredKinFuImpl<UMat>::update(InputArray _depth, InputArray _rgb)
{
    Size frameSize(volumeSettings.getWidth(), volumeSettings.getHeight());
    CV_Assert(!_depth.empty() && _depth.size() == frameSize);

    UMat depth;
    UMat rgb;
    if(!_depth.isUMat())
    {
        _depth.copyTo(depth);
        _rgb.copyTo(rgb);
        return updateT(depth, rgb);
    }
    else
    {
        return updateT(_depth.getUMat(), _rgb.getUMat());
    }
}


template< typename MatType >
bool ColoredKinFuImpl<MatType>::updateT(const MatType& _depth, const MatType& _rgb)
{
    CV_TRACE_FUNCTION();

    MatType depth, rgb;

    if(_depth.type() != DEPTH_TYPE)
        _depth.convertTo(depth, DEPTH_TYPE);
    else
        depth = _depth;

    if (_rgb.type() != COLOR_TYPE)
    {
        MatType rgb_tmp;
        std::vector<MatType> channels;
        _rgb.convertTo(rgb_tmp, COLOR_TYPE);
        cv::split(rgb_tmp, channels);
        channels.push_back(MatType::zeros(channels[0].size(), CV_32F));
        merge(channels, rgb);
    }
    else
        rgb = _rgb;

    OdometryFrame newFrame = icp.createOdometryFrame();

    newFrame.setImage(rgb);
    newFrame.setDepth(depth);

    if(frameCounter == 0)
    {
        icp.prepareFrame(newFrame);

        // use depth instead of distance
        volume.integrate(depth, rgb, pose);
        // TODO: try to move setPyramidLevel from kinfu to volume
        newFrame.setPyramidLevel(icpIterations.size(), OdometryFramePyramidType::PYR_IMAGE);
        newFrame.setPyramidAt(rgb, OdometryFramePyramidType::PYR_IMAGE, 0);
    }
    else
    {
        Affine3f affine;
        Matx44d mrt;
        Mat Rt;
        icp.prepareFrames(newFrame, prevFrame);
        bool success = icp.compute(newFrame, prevFrame, Rt);

        if (!success)
            return false;
        affine.matrix = Matx44f(Rt);
        pose = (Affine3f(pose) * affine).matrix;

        float rnorm = (float)cv::norm(affine.rvec());
        float tnorm = (float)cv::norm(affine.translation());
        // We do not integrate volume if camera does not move
        if((rnorm + tnorm)/2 >= tsdf_min_camera_movement)
        {
            // use depth instead of distance
            volume.integrate(depth, rgb, pose);
            newFrame.setPyramidLevel(icpIterations.size(), OdometryFramePyramidType::PYR_IMAGE);
            newFrame.setPyramidAt(rgb, OdometryFramePyramidType::PYR_IMAGE, 0);
        }
        MatType points, normals, colors;
        newFrame.getPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
        newFrame.getPyramidAt(normals, OdometryFramePyramidType::PYR_NORM,  0);
        newFrame.getPyramidAt(colors, OdometryFramePyramidType::PYR_IMAGE, 0);

        volume.raycast(pose, volumeSettings.getHeight(), volumeSettings.getWidth(), points, normals, colors);

        newFrame.setPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
        newFrame.setPyramidAt(normals, OdometryFramePyramidType::PYR_NORM,  0);
        newFrame.setPyramidAt(colors, OdometryFramePyramidType::PYR_IMAGE, 0);
    }

    renderFrame = newFrame;
    prevFrame = newFrame;

    frameCounter++;
    return true;
}


template< typename MatType >
void ColoredKinFuImpl<MatType>::render(OutputArray image) const
{
    CV_TRACE_FUNCTION();
    MatType pts, nrm, rgb;
    renderFrame.getPyramidAt(pts, OdometryFramePyramidType::PYR_CLOUD, 0);
    renderFrame.getPyramidAt(nrm, OdometryFramePyramidType::PYR_NORM, 0);
    renderFrame.getPyramidAt(rgb, OdometryFramePyramidType::PYR_IMAGE, 0);

    detail::renderPointsNormalsColors(pts, nrm, rgb, image);
}

template< typename MatType >
void ColoredKinFuImpl<MatType>::render(OutputArray image, const Matx44f& _cameraPose) const
{
    CV_TRACE_FUNCTION();

    MatType points, normals, colors;
    volume.raycast(_cameraPose, volumeSettings.getHeight(), volumeSettings.getWidth(), points, normals, colors);
    detail::renderPointsNormalsColors(points, normals, colors, image);
}


template< typename MatType >
void ColoredKinFuImpl<MatType>::getCloud(OutputArray p, OutputArray n) const
{
    volume.fetchPointsNormals(p, n);
}


template< typename MatType >
void ColoredKinFuImpl<MatType>::getPoints(OutputArray points) const
{
    volume.fetchPointsNormals(points, noArray());
}


template< typename MatType >
void ColoredKinFuImpl<MatType>::getNormals(InputArray points, OutputArray normals) const
{
    volume.fetchNormals(points, normals);
}

// importing class

#ifdef OPENCV_ENABLE_NONFREE

Ptr<ColoredKinFu> ColoredKinFu::create()
{
    return makePtr< ColoredKinFuImpl<Mat> >();
}

#else
Ptr<ColoredKinFu> ColoredKinFu::create()
{
    CV_Error(Error::StsNotImplemented,
             "This algorithm is patented and is excluded in this configuration; "
             "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}
#endif

ColoredKinFu::~ColoredKinFu() {}

} // namespace kinfu
} // namespace cv
