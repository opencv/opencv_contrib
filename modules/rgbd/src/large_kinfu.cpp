// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#include "precomp.hpp"

namespace cv
{
namespace large_kinfu
{
using namespace kinfu;


// MatType should be Mat or UMat
template<typename MatType>
class LargeKinfuImpl : public LargeKinfu
{
   public:
    LargeKinfuImpl();
    virtual ~LargeKinfuImpl();

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
    VolumeSettings volumeSettings;
    OdometrySettings ods;

    Odometry icp;
    //! TODO: Submap manager and Pose graph optimizer
    cv::Ptr<detail::SubmapManager<MatType>> submapMgr;

    int frameCounter;
    Affine3f pose;

    float tsdf_min_camera_movement = 0.f; //meters, disabled
    Vec3f lightPose = Vec3f::all(0.f);
};

template<typename MatType>
LargeKinfuImpl<MatType>::LargeKinfuImpl()
{
    volumeSettings = VolumeSettings(VolumeType::HashTSDF);

    Matx33f intr;
    volumeSettings.getCameraIntrinsics(intr);
    Vec3i res;
    volumeSettings.getVolumeResolution(res);

    ods = OdometrySettings();
    ods.setCameraMatrix(intr);
    ods.setMaxRotation(30.f);
    ods.setMaxTranslation(volumeSettings.getVoxelSize() * res[0] * 0.5f);
    icp = Odometry(OdometryType::DEPTH, ods, OdometryAlgoType::FAST);

    submapMgr = cv::makePtr<detail::SubmapManager<MatType>>(volumeSettings);
    reset();
    submapMgr->createNewSubmap(true);
}

template<typename MatType>
void LargeKinfuImpl<MatType>::reset()
{
    frameCounter = 0;
    pose         = Affine3f::Identity();
    submapMgr->reset();
}

template<typename MatType>
LargeKinfuImpl<MatType>::~LargeKinfuImpl()
{
}

template<typename MatType>
const Affine3f LargeKinfuImpl<MatType>::getPose() const
{
    return pose;
}

template<>
bool LargeKinfuImpl<Mat>::update(InputArray _depth)
{
    Size frameSize(volumeSettings.getWidth(), volumeSettings.getHeight());
    CV_Assert(!_depth.empty() && _depth.size() == frameSize);

    Mat depth;
    if (_depth.isUMat())
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
bool LargeKinfuImpl<UMat>::update(InputArray _depth)
{
    Size frameSize(volumeSettings.getWidth(), volumeSettings.getHeight());
    CV_Assert(!_depth.empty() && _depth.size() == frameSize);

    UMat depth;
    if (!_depth.isUMat())
    {
        _depth.copyTo(depth);
        return updateT(depth);
    }
    else
    {
        return updateT(_depth.getUMat());
    }
}


template<typename MatType>
bool LargeKinfuImpl<MatType>::updateT(const MatType& _depth)
{
    CV_TRACE_FUNCTION();

    Size frameSize(volumeSettings.getWidth(), volumeSettings.getHeight());
    
    Matx33f intr;
    volumeSettings.getCameraIntrinsics(intr);

    MatType depth;
    if (_depth.type() != DEPTH_TYPE)
        _depth.convertTo(depth, DEPTH_TYPE);
    else
        depth = _depth;

    OdometryFrame newFrame = icp.createOdometryFrame();
    newFrame.setDepth(depth);

    CV_LOG_INFO(NULL, "Current frameID: " << frameCounter);
    for (const auto& it : submapMgr->activeSubmaps)
    {
        int currTrackingId = it.first;
        auto submapData = it.second;
        Ptr<detail::Submap<MatType>> currTrackingSubmap = submapMgr->getSubmap(currTrackingId);
        Affine3f affine;
        CV_LOG_INFO(NULL, "Current tracking ID: " << currTrackingId);

        if(frameCounter == 0) //! Only one current tracking map
        {
            icp.prepareFrame(newFrame);
            currTrackingSubmap->integrate(depth, frameCounter);
            currTrackingSubmap->frame = newFrame;
            currTrackingSubmap->renderFrame = newFrame;
            continue;
        }

        //1. Track
        Matx44d mrt;
        Mat Rt;
        icp.prepareFrames(newFrame, currTrackingSubmap->frame);
        bool trackingSuccess = icp.compute(newFrame, currTrackingSubmap->frame, Rt);

        if (trackingSuccess)
        {
            affine.matrix = Matx44f(Rt);
            currTrackingSubmap->composeCameraPose(affine);
        }
        else
        {
            CV_LOG_INFO(NULL, "Tracking failed");
            continue;
        }

        //2. Integrate
        if(submapData.type == detail::SubmapManager<MatType>::Type::NEW || submapData.type == detail::SubmapManager<MatType>::Type::CURRENT)
        {
            float rnorm = (float)cv::norm(affine.rvec());
            float tnorm = (float)cv::norm(affine.translation());
            // We do not integrate volume if camera does not move
            if ((rnorm + tnorm) / 2 >= tsdf_min_camera_movement)
                currTrackingSubmap->integrate(depth, frameCounter);
        }

        //3. Raycast
        currTrackingSubmap->raycast(this->icp, currTrackingSubmap->cameraPose, intr, frameSize);

        CV_LOG_INFO(NULL, "Submap: " << currTrackingId << " Total allocated blocks: " << currTrackingSubmap->getTotalAllocatedBlocks());
        CV_LOG_INFO(NULL, "Submap: " << currTrackingId << " Visible blocks: " << currTrackingSubmap->getVisibleBlocks(frameCounter));

    }
    //4. Update map
    bool isMapUpdated = submapMgr->updateMap(frameCounter, newFrame);

    if(isMapUpdated)
    {
        // TODO: Convert constraints to posegraph
        Ptr<detail::PoseGraph> poseGraph = submapMgr->MapToPoseGraph();
        CV_LOG_INFO(NULL, "Created posegraph");
        LevMarq::Report r = poseGraph->optimize();
        if (!r.found)
        {
            CV_LOG_INFO(NULL, "Failed to perform pose graph optimization");
            return false;
        }

        submapMgr->PoseGraphToMap(poseGraph);

    }
    CV_LOG_INFO(NULL, "Number of submaps: " << submapMgr->submapList.size());

    frameCounter++;
    return true;
}


template<typename MatType>
void LargeKinfuImpl<MatType>::render(OutputArray image) const
{
    CV_TRACE_FUNCTION();
    auto currSubmap = submapMgr->getCurrentSubmap();
    //! TODO: Can render be dependent on current submap
    MatType pts, nrm;
    currSubmap->renderFrame.getPyramidAt(pts, OdometryFramePyramidType::PYR_CLOUD, 0);
    currSubmap->renderFrame.getPyramidAt(nrm, OdometryFramePyramidType::PYR_NORM,  0);
    detail::renderPointsNormals(pts, nrm, image, lightPose);
}


template<typename MatType>
void LargeKinfuImpl<MatType>::render(OutputArray image, const Matx44f& _cameraPose) const
{
    CV_TRACE_FUNCTION();

    Size frameSize(volumeSettings.getWidth(), volumeSettings.getHeight());
    Affine3f cameraPose(_cameraPose);
    auto currSubmap = submapMgr->getCurrentSubmap();
    Matx33f intr;
    volumeSettings.getCameraIntrinsics(intr);
    MatType points, normals;
    currSubmap->raycast(this->icp, cameraPose, intr, frameSize, points, normals);
    detail::renderPointsNormals(points, normals, image, lightPose);
}


template<typename MatType>
void LargeKinfuImpl<MatType>::getCloud(OutputArray p, OutputArray n) const
{
    auto currSubmap = submapMgr->getCurrentSubmap();
    currSubmap->volume.fetchPointsNormals(p, n);
}

template<typename MatType>
void LargeKinfuImpl<MatType>::getPoints(OutputArray points) const
{
    auto currSubmap = submapMgr->getCurrentSubmap();
    currSubmap->volume.fetchPointsNormals(points, noArray());
}

template<typename MatType>
void LargeKinfuImpl<MatType>::getNormals(InputArray points, OutputArray normals) const
{
    auto currSubmap = submapMgr->getCurrentSubmap();
    currSubmap->volume.fetchNormals(points, normals);
}

// importing class

#ifdef OPENCV_ENABLE_NONFREE

Ptr<LargeKinfu> LargeKinfu::create()
{
#ifdef HAVE_OPENCL
    if (cv::ocl::useOpenCL())
        return makePtr<LargeKinfuImpl<UMat>>();
#endif
    return makePtr<LargeKinfuImpl<Mat>>();
}

#else
Ptr<LargeKinfu> LargeKinfu::create()
{
    CV_Error(Error::StsNotImplemented,
             "This algorithm is patented and is excluded in this configuration; "
             "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}
#endif
}  // namespace large_kinfu
}  // namespace cv
