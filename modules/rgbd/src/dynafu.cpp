// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "dynafu_tsdf.hpp"
#include "warpfield.hpp"

#include "fast_icp.hpp"
#include "kinfu_frame.hpp"

#include "opencv2/core/opengl.hpp"

#ifdef HAVE_OPENGL
#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
# include <OpenGL/gl.h>
#else
#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#endif
# include <GL/gl.h>
#endif
#endif

namespace cv {
namespace dynafu {
using namespace kinfu;

Ptr<Params> Params::defaultParams()
{
    Params p;

    p.frameSize = Size(640, 480);

    float fx, fy, cx, cy;
    fx = fy = 525.f;
    cx = p.frameSize.width/2 - 0.5f;
    cy = p.frameSize.height/2 - 0.5f;
    p.intr = Matx33f(fx,  0, cx,
                      0, fy, cy,
                      0,  0,  1);

    // 5000 for the 16-bit PNG files
    // 1 for the 32-bit float images in the ROS bag files
    p.depthFactor = 5000;

    // sigma_depth is scaled by depthFactor when calling bilateral filter
    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    p.icpAngleThresh = (float)(30. * CV_PI / 180.); // radians
    p.icpDistThresh = 0.1f; // meters

    p.icpIterations = {10, 5, 4};
    p.pyramidLevels = (int)p.icpIterations.size();

    p.tsdf_min_camera_movement = 0.f; //meters, disabled

    p.volumeDims = Vec3i::all(512); //number of voxels

    float volSize = 3.f;
    p.voxelSize = volSize/512.f; //meters

    // default pose of volume cube
    p.volumePose = Affine3f().translate(Vec3f(-volSize/2.f, -volSize/2.f, 0.5f));
    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.25f;  //in voxel sizes
    // gradient delta factor is fixed at 1.0f and is not used
    //p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.lightPose = p.volume_pose.translation()/4; //meters
    p.lightPose = Vec3f::all(0.f); //meters

    // depth truncation is not used by default but can be useful in some scenes
    p.truncateThreshold = 0.f; //meters

    return makePtr<Params>(p);
}

Ptr<Params> Params::coarseParams()
{
    Ptr<Params> p = defaultParams();

    p->icpIterations = {5, 3, 2};
    p->pyramidLevels = (int)p->icpIterations.size();

    float volSize = 3.f;
    p->volumeDims = Vec3i::all(128); //number of voxels
    p->voxelSize  = volSize/128.f;

    p->raycast_step_factor = 0.75f;  //in voxel sizes

    return p;
}

// T should be Mat or UMat
template< typename T >
class DynaFuImpl : public DynaFu
{
public:
    DynaFuImpl(const Params& _params);
    virtual ~DynaFuImpl();

    const Params& getParams() const CV_OVERRIDE;

    void render(OutputArray image, const Matx44f& cameraPose) const CV_OVERRIDE;

    void getCloud(OutputArray points, OutputArray normals) const CV_OVERRIDE;
    void getPoints(OutputArray points) const CV_OVERRIDE;
    void getNormals(InputArray points, OutputArray normals) const CV_OVERRIDE;

    void reset() CV_OVERRIDE;

    const Affine3f getPose() const CV_OVERRIDE;

    bool update(InputArray depth) CV_OVERRIDE;

    bool updateT(const T& depth);

    std::vector<Point3f> getNodesPos() const CV_OVERRIDE;

    void marchCubes(OutputArray vertices, OutputArray edges) const CV_OVERRIDE;

    void renderSurface(OutputArray image) CV_OVERRIDE;

private:
    Params params;

    cv::Ptr<ICP> icp;
    cv::Ptr<TSDFVolume> volume;

    int frameCounter;
    Affine3f pose;
    std::vector<T> pyrPoints;
    std::vector<T> pyrNormals;

    WarpField warpfield;

    ogl::Arrays arr;
    ogl::Buffer idx;

    void drawScene(OutputArray img);
};

template< typename T>
std::vector<Point3f> DynaFuImpl<T>::getNodesPos() const {
    NodeVectorType nv = warpfield.getNodes();
    std::vector<Point3f> nodesPos(nv.size());
    for(auto n: nv)
        nodesPos.push_back(n->pos);

    return nodesPos;
}

template< typename T >
DynaFuImpl<T>::DynaFuImpl(const Params &_params) :
    params(_params),
    icp(makeICP(params.intr, params.icpIterations, params.icpAngleThresh, params.icpDistThresh)),
    volume(makeTSDFVolume(params.volumeDims, params.voxelSize, params.volumePose,
                          params.tsdf_trunc_dist, params.tsdf_max_weight,
                          params.raycast_step_factor)),
    pyrPoints(), pyrNormals(), warpfield()
{
#ifdef HAVE_OPENGL
    // Bind framebuffer for off-screen rendering
    unsigned int fbo_depth;
    glGenRenderbuffersEXT(1, &fbo_depth);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, fbo_depth);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, params.frameSize.width, params.frameSize.height);

    unsigned int fbo;
    glGenFramebuffersEXT(1, &fbo);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);

    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, fbo_depth);
#endif

    reset();
}

template< typename T >
void DynaFuImpl<T>::drawScene(OutputArray image)
{
#ifdef HAVE_OPENGL
    glViewport(0, 0, params.frameSize.width, params.frameSize.height);

    glEnable(GL_DEPTH_TEST);
    glClear(GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    float fovX = params.frameSize.width/params.intr(0, 0);
    float fovY = params.frameSize.height/params.intr(1, 1);

    Vec3f t;
    t = params.volumePose.translation();

    double nearZ = t[2];
    double farZ = params.volumeDims[2] * params.voxelSize + nearZ;

    // Define viewing volume
    glFrustum(-nearZ*fovX/2, nearZ*fovX/2, -nearZ*fovY/2, nearZ*fovY/2, nearZ, farZ);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(1.f, 1.f, -1.f); //Flip Z as camera points towards -ve Z axis

    ogl::render(arr, idx, ogl::TRIANGLES);

    float f[params.frameSize.width*params.frameSize.height];
    glReadPixels(0, 0, params.frameSize.width, params.frameSize.height, GL_DEPTH_COMPONENT, GL_FLOAT, &f[0]);

    Mat depthData;
    Mat(params.frameSize.height, params.frameSize.width, CV_32F, f).convertTo(depthData, CV_8U, 255);
    depthData.copyTo(image);
#else
    CV_Error(cv::Error::StsBadFunc, "OpenGL support not enabled. Please rebuild the library with OpenGL support");
#endif
}

template< typename T >
void DynaFuImpl<T>::reset()
{
    frameCounter = 0;
    pose = Affine3f::Identity();
    volume->reset();
}

template< typename T >
DynaFuImpl<T>::~DynaFuImpl()
{ }

template< typename T >
const Params& DynaFuImpl<T>::getParams() const
{
    return params;
}

template< typename T >
const Affine3f DynaFuImpl<T>::getPose() const
{
    return pose;
}


template<>
bool DynaFuImpl<Mat>::update(InputArray _depth)
{
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

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
bool DynaFuImpl<UMat>::update(InputArray _depth)
{
    CV_TRACE_FUNCTION();

    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

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


template< typename T >
bool DynaFuImpl<T>::updateT(const T& _depth)
{
    CV_TRACE_FUNCTION();

    T depth;
    if(_depth.type() != DEPTH_TYPE)
        _depth.convertTo(depth, DEPTH_TYPE);
    else
        depth = _depth;

    std::vector<T> newPoints, newNormals;
    makeFrameFromDepth(depth, newPoints, newNormals, params.intr,
                       params.pyramidLevels,
                       params.depthFactor,
                       params.bilateral_sigma_depth,
                       params.bilateral_sigma_spatial,
                       params.bilateral_kernel_size,
                       params.truncateThreshold);

    if(frameCounter == 0)
    {
        // use depth instead of distance
        volume->integrate(depth, params.depthFactor, pose, params.intr, makePtr<WarpField>(warpfield));

        pyrPoints  = newPoints;
        pyrNormals = newNormals;
    }
    else
    {

        UMat wfPoints;
        UMat wfNormals;
        volume->fetchPointsNormals(wfPoints, wfNormals);
        warpfield.updateNodesFromPoints(wfPoints);

        Affine3f affine;
        bool success = icp->estimateTransform(affine, pyrPoints, pyrNormals, newPoints, newNormals);
        if(!success)
            return false;

        pose = pose * affine;
        warpfield.setAllRT(Affine3f::Identity());

        float rnorm = (float)cv::norm(affine.rvec());
        float tnorm = (float)cv::norm(affine.translation());
        // We do not integrate volume if camera does not move
        if((rnorm + tnorm)/2 >= params.tsdf_min_camera_movement)
        {
            // use depth instead of distance
            volume->integrate(depth, params.depthFactor, pose, params.intr, makePtr<WarpField>(warpfield));
        }

        T& points  = pyrPoints [0];
        T& normals = pyrNormals[0];
        volume->raycast(pose, params.intr, params.frameSize, points, normals);
        // build a pyramid of points and normals
        buildPyramidPointsNormals(points, normals, pyrPoints, pyrNormals,
                                  params.pyramidLevels);
    }

    std::cout << "Frame# " << frameCounter++ << std::endl;
    return true;
}


template< typename T >
void DynaFuImpl<T>::render(OutputArray image, const Matx44f& _cameraPose) const
{
    CV_TRACE_FUNCTION();

    Affine3f cameraPose(_cameraPose);

    const Affine3f id = Affine3f::Identity();
    if((cameraPose.rotation() == pose.rotation() && cameraPose.translation() == pose.translation()) ||
       (cameraPose.rotation() == id.rotation()   && cameraPose.translation() == id.translation()))
    {
        renderPointsNormals(pyrPoints[0], pyrNormals[0], image, params.lightPose);
    }
    else
    {
        T points, normals;
        volume->raycast(cameraPose, params.intr, params.frameSize, points, normals);
        renderPointsNormals(points, normals, image, params.lightPose);
    }
}


template< typename T >
void DynaFuImpl<T>::getCloud(OutputArray p, OutputArray n) const
{
    volume->fetchPointsNormals(p, n);
}


template< typename T >
void DynaFuImpl<T>::getPoints(OutputArray points) const
{
    volume->fetchPointsNormals(points, noArray());
}


template< typename T >
void DynaFuImpl<T>::getNormals(InputArray points, OutputArray normals) const
{
    volume->fetchNormals(points, normals);
}

template< typename T >
void DynaFuImpl<T>::marchCubes(OutputArray vertices, OutputArray edges) const
{
    volume->marchCubes(vertices, edges);
}

template<typename T>
void DynaFuImpl<T>::renderSurface(OutputArray image)
{
    Mat vertices, meshIdx;
    volume->marchCubes(vertices, noArray());
    if(vertices.empty()) return;

    Affine3f invCamPose(pose.inv());
    for(int i = 0; i < vertices.size().height; i++) {
        Vec4f v = vertices.at<Vec4f>(i);
        Point3f p = invCamPose * Point3f(v[0], v[1], v[2]);
        vertices.at<Vec4f>(i) = Vec4f(p.x, p.y, p.z, 1.f);
    }

    for(int i = 0; i < vertices.size().height; i++)
        meshIdx.push_back<int>(i);

    arr.setVertexArray(vertices);
    idx.copyFrom(meshIdx);

    drawScene(image);
}

// importing class

#ifdef OPENCV_ENABLE_NONFREE

Ptr<DynaFu> DynaFu::create(const Ptr<Params>& params)
{
    return makePtr< DynaFuImpl<Mat> >(*params);
}

#else
Ptr<DynaFu> DynaFu::create(const Ptr<Params>& /*params*/)
{
    CV_Error(Error::StsNotImplemented,
             "This algorithm is patented and is excluded in this configuration; "
             "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}
#endif

DynaFu::~DynaFu() {}

} // namespace dynafu
} // namespace cv
