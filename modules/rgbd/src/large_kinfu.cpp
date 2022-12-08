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

Ptr<VolumeParams> VolumeParams::defaultParams(VolumeType _volumeKind)
{
    VolumeParams params;
    params.kind = _volumeKind;
    params.maxWeight = 64;
    params.raycastStepFactor = 0.25f;
    params.unitResolution = 0;  // unitResolution not used for TSDF
    float volumeSize = 3.0f;
    Matx44f pose = Affine3f().translate(Vec3f(-volumeSize / 2.f, -volumeSize / 2.f, 0.5f)).matrix;
    params.pose = Mat(pose);

    if (params.kind == VolumeType::TSDF)
    {
        params.resolutionX = 512;
        params.resolutionY = 512;
        params.resolutionZ = 512;
        params.voxelSize = volumeSize / 512.f;
        params.depthTruncThreshold = 0.f;  // depthTruncThreshold not required for TSDF
        params.tsdfTruncDist = 7 * params.voxelSize;  // About 0.04f in meters
        return makePtr<VolumeParams>(params);
    }
    else if (params.kind == VolumeType::HashTSDF)
    {
        params.unitResolution = 16;
        params.voxelSize = volumeSize / 512.f;
        params.depthTruncThreshold = 4.f;
        params.tsdfTruncDist = 7 * params.voxelSize;  // About 0.04f in meters
        return makePtr<VolumeParams>(params);
    }
    else if (params.kind == VolumeType::ColorTSDF)
    {
        params.resolutionX = 512;
        params.resolutionY = 512;
        params.resolutionZ = 512;
        params.voxelSize = volumeSize / 512.f;
        params.depthTruncThreshold = 0.f;  // depthTruncThreshold not required for TSDF
        params.tsdfTruncDist = 7 * params.voxelSize;  // About 0.04f in meters
        return makePtr<VolumeParams>(params);
    }
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

Ptr<VolumeParams> VolumeParams::coarseParams(VolumeType _volumeKind)
{
    Ptr<VolumeParams> params = defaultParams(_volumeKind);

    params->raycastStepFactor = 0.75f;
    float volumeSize = 3.0f;
    if (params->kind == VolumeType::TSDF)
    {
        params->resolutionX = 128;
        params->resolutionY = 128;
        params->resolutionZ = 128;
        params->voxelSize = volumeSize / 128.f;
        params->tsdfTruncDist = 2 * params->voxelSize;  // About 0.04f in meters
        return params;
    }
    else if (params->kind == VolumeType::HashTSDF)
    {
        params->voxelSize = volumeSize / 128.f;
        params->tsdfTruncDist = 2 * params->voxelSize;  // About 0.04f in meters
        return params;
    }
    else if (params->kind == VolumeType::ColorTSDF)
    {
        params->resolutionX = 128;
        params->resolutionY = 128;
        params->resolutionZ = 128;
        params->voxelSize = volumeSize / 128.f;
        params->tsdfTruncDist = 2 * params->voxelSize;  // About 0.04f in meters
        return params;
    }
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

Ptr<Params> Params::defaultParams()
{
    Params p;

    // Frame parameters
    {
        p.frameSize = Size(640, 480);

        float fx, fy, cx, cy;
        fx = fy = 525.f;
        cx      = p.frameSize.width / 2.0f - 0.5f;
        cy      = p.frameSize.height / 2.0f - 0.5f;
        p.intr  = Matx33f(fx, 0, cx, 0, fy, cy, 0, 0, 1);

        // 5000 for the 16-bit PNG files
        // 1 for the 32-bit float images in the ROS bag files
        p.depthFactor = 5000;

        // sigma_depth is scaled by depthFactor when calling bilateral filter
        p.bilateral_sigma_depth   = 0.04f;  // meter
        p.bilateral_sigma_spatial = 4.5;    // pixels
        p.bilateral_kernel_size   = 7;      // pixels
        p.truncateThreshold       = 0.f;    // meters
    }
    // ICP parameters
    {
        p.icpAngleThresh = (float)(30. * CV_PI / 180.);  // radians
        p.icpDistThresh  = 0.1f;                         // meters

        p.icpIterations = { 10, 5, 4 };
        p.pyramidLevels = (int)p.icpIterations.size();
    }
    // Volume parameters
    {
        float volumeSize                   = 3.0f;
        p.volumeParams.kind                = VolumeType::TSDF;
        p.volumeParams.resolutionX         = 512;
        p.volumeParams.resolutionY         = 512;
        p.volumeParams.resolutionZ         = 512;
        Affine3f newPose = Affine3f().translate(Vec3f(-volumeSize / 2.f, -volumeSize / 2.f, 0.5f));
        p.volumeParams.pose                = Mat(newPose.matrix);
        p.volumeParams.voxelSize           = volumeSize / 512.f;            // meters
        p.volumeParams.tsdfTruncDist       = 7 * p.volumeParams.voxelSize;  // about 0.04f in meters
        p.volumeParams.maxWeight           = 64;                            // frames
        p.volumeParams.raycastStepFactor   = 0.25f;                         // in voxel sizes
        p.volumeParams.depthTruncThreshold = p.truncateThreshold;
    }
    // Unused parameters
    p.tsdf_min_camera_movement = 0.f;              // meters, disabled
    p.lightPose                = Vec3f::all(0.f);  // meters

    return makePtr<Params>(p);
}

Ptr<Params> Params::coarseParams()
{
    Ptr<Params> p = defaultParams();

    // Reduce ICP iterations and pyramid levels
    {
        p->icpIterations = { 5, 3, 2 };
        p->pyramidLevels = (int)p->icpIterations.size();
    }
    // Make the volume coarse
    {
        float volumeSize                  = 3.f;
        p->volumeParams.resolutionX       = 128;  // number of voxels
        p->volumeParams.resolutionY       = 128;
        p->volumeParams.resolutionZ       = 128;
        p->volumeParams.voxelSize         = volumeSize / 128.f;
        p->volumeParams.tsdfTruncDist     = 2 * p->volumeParams.voxelSize;  // 0.04f in meters
        p->volumeParams.raycastStepFactor = 0.75f;                          // in voxel sizes
    }
    return p;
}
Ptr<Params> Params::hashTSDFParams(bool isCoarse)
{
    Ptr<Params> p;
    if (isCoarse)
        p = coarseParams();
    else
        p = defaultParams();

    p->volumeParams.kind                = VolumeType::HashTSDF;
    p->volumeParams.depthTruncThreshold = 4.f;
    p->volumeParams.unitResolution      = 16;
    return p;
}

// MatType should be Mat or UMat
template<typename MatType>
class LargeKinfuImpl : public LargeKinfu
{
   public:
    LargeKinfuImpl(const Params& _params);
    virtual ~LargeKinfuImpl();

    static VolumeSettings paramsToSettings(const Params& params);
    const Params& getParams() const CV_OVERRIDE;

    void render(OutputArray image) const CV_OVERRIDE;
    void render(OutputArray image, const Matx44f& cameraPose) const CV_OVERRIDE;

    virtual void getCloud(OutputArray points, OutputArray normals) const CV_OVERRIDE;
    void getPoints(OutputArray points) const CV_OVERRIDE;
    void getNormals(InputArray points, OutputArray normals) const CV_OVERRIDE;

    void reset() CV_OVERRIDE;

    Affine3f getPose() const CV_OVERRIDE;

    bool update(InputArray depth) CV_OVERRIDE;

    bool updateT(const MatType& depth);

   private:
    Params params;
    VolumeSettings settings;

    Odometry icp;
    // TODO: Submap manager and Pose graph optimizer
    cv::Ptr<detail::SubmapManager<MatType>> submapMgr;

    int frameCounter;
    Affine3f pose;
};

template<typename MatType>
VolumeSettings LargeKinfuImpl<MatType>::paramsToSettings(const Params& params)
{
    VolumeSettings vs(VolumeType::HashTSDF);
    vs.setMaxDepth(params.truncateThreshold);
    vs.setCameraIntegrateIntrinsics(params.intr);
    vs.setCameraRaycastIntrinsics(params.intr);
    vs.setDepthFactor(params.depthFactor);

    vs.setVoxelSize(params.volumeParams.voxelSize);
    vs.setVolumePose(params.volumeParams.pose);
    vs.setRaycastStepFactor(params.volumeParams.raycastStepFactor);
    vs.setTsdfTruncateDistance(params.volumeParams.tsdfTruncDist);
    vs.setMaxWeight(params.volumeParams.maxWeight);
    vs.setVolumeResolution(Vec3i(params.volumeParams.unitResolution, params.volumeParams.unitResolution, params.volumeParams.unitResolution));

    return vs;
}

template<typename MatType>
LargeKinfuImpl<MatType>::LargeKinfuImpl(const Params& _params)
    : params(_params),
    settings(paramsToSettings(params))
{
    OdometrySettings ods;
    ods.setCameraMatrix(Mat(params.intr));
    ods.setMaxRotation(30.f);
    ods.setMaxTranslation(params.volumeParams.voxelSize * params.volumeParams.resolutionX * 0.5f);
    icp = Odometry(OdometryType::DEPTH, ods, OdometryAlgoType::FAST);

    submapMgr = cv::makePtr<detail::SubmapManager<MatType>>(settings);
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
const Params& LargeKinfuImpl<MatType>::getParams() const
{
    return params;
}

template<typename MatType>
Affine3f LargeKinfuImpl<MatType>::getPose() const
{
    return pose;
}

template<>
bool LargeKinfuImpl<Mat>::update(InputArray _depth)
{
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

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
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

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

    MatType depth;
    if (_depth.type() != DEPTH_TYPE)
        _depth.convertTo(depth, DEPTH_TYPE);
    else
        depth = _depth;

    OdometryFrame newFrame(depth);

    CV_LOG_INFO(NULL, "Current frameID: " << frameCounter);
    for (const auto& it : submapMgr->activeSubmaps)
    {
        int currTrackingId = it.first;
        auto submapData = it.second;
        Ptr<detail::Submap<MatType>> currTrackingSubmap = submapMgr->getSubmap(currTrackingId);
        Affine3f affine;
        CV_LOG_INFO(NULL, "Current tracking ID: " << currTrackingId);

        if(frameCounter == 0) // Only one current tracking map
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
            if ((rnorm + tnorm) / 2 >= params.tsdf_min_camera_movement)
                currTrackingSubmap->integrate(depth, frameCounter);
        }

        //3. Raycast
        currTrackingSubmap->raycast(currTrackingSubmap->cameraPose, params.frameSize, params.intr);

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
    // TODO: Can render be dependent on current submap
    MatType pts, nrm;
    currSubmap->renderFrame.getPyramidAt(pts, OdometryFramePyramidType::PYR_CLOUD, 0);
    currSubmap->renderFrame.getPyramidAt(nrm, OdometryFramePyramidType::PYR_NORM,  0);
    detail::renderPointsNormals(pts, nrm, image, params.lightPose);
}


template<typename MatType>
void LargeKinfuImpl<MatType>::render(OutputArray image, const Matx44f& _cameraPose) const
{
    CV_TRACE_FUNCTION();

    Affine3f cameraPose(_cameraPose);
    auto currSubmap = submapMgr->getCurrentSubmap();
    MatType points, normals;
    currSubmap->raycast(cameraPose, params.frameSize, params.intr, points, normals);
    detail::renderPointsNormals(points, normals, image, params.lightPose);
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

Ptr<LargeKinfu> LargeKinfu::create(const Ptr<Params>& params)
{
    CV_Assert((int)params->icpIterations.size() == params->pyramidLevels);
    CV_Assert(params->intr(0, 1) == 0 && params->intr(1, 0) == 0 && params->intr(2, 0) == 0 && params->intr(2, 1) == 0 &&
              params->intr(2, 2) == 1);
#ifdef HAVE_OPENCL
    if (cv::ocl::useOpenCL())
        return makePtr<LargeKinfuImpl<UMat>>(*params);
#endif
    return makePtr<LargeKinfuImpl<Mat>>(*params);
}

#else
Ptr<LargeKinfu> LargeKinfu::create(const Ptr<Params>& /* params */)
{
    CV_Error(Error::StsNotImplemented,
             "This algorithm is patented and is excluded in this configuration; "
             "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}
#endif
}  // namespace large_kinfu
}  // namespace cv
