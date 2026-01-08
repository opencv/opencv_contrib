// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "dynafu_tsdf.hpp"
#include "warpfield.hpp"

#include "fast_icp.hpp"
#include "nonrigid_icp.hpp"
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

// GL Extention definitions missing from standard Win32 gl.h
#if defined(_WIN32) && !defined(GL_RENDERBUFFER_EXT)
#define GL_COLOR_ATTACHMENT0_EXT 0x8CE0
#define GL_DEPTH_ATTACHMENT_EXT 0x8D00
#define GL_FRAMEBUFFER_EXT 0x8D40
#define GL_RENDERBUFFER_EXT 0x8D41
namespace {
PROC _wglGetProcAddress(const char *name)
{
  auto proc = wglGetProcAddress(name);
  if (!proc)
    CV_Error(cv::Error::OpenGlApiCallError, cv::format("Can't load OpenGL extension [%s]", name) );
  return proc;
}

void glGenFramebuffersEXT(GLsizei n, GLuint *framebuffers)
{
  static auto proc = reinterpret_cast<void(*)(GLsizei, GLuint*)>(_wglGetProcAddress(__func__));
  proc(n, framebuffers);
}
void glGenRenderbuffersEXT(GLsizei n, GLuint *renderbuffers)
{
  static auto proc = reinterpret_cast<void(*)(GLsizei, GLuint*)>(_wglGetProcAddress(__func__));
  proc(n, renderbuffers);
}
void glBindRenderbufferEXT(GLenum target, GLuint renderbuffer)
{
  static auto proc = reinterpret_cast<void(*)(GLenum, GLuint)>(_wglGetProcAddress(__func__));
  proc(target, renderbuffer);
}
void glBindFramebufferEXT(GLenum target, GLuint framebuffer)
{
  static auto proc = reinterpret_cast<void(*)(GLenum, GLuint)>(_wglGetProcAddress(__func__));
  proc(target, framebuffer);
}
void glFramebufferRenderbufferEXT(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer)
{
  static auto proc = reinterpret_cast<void(*)(GLenum, GLenum, GLenum, GLuint)>(_wglGetProcAddress(__func__));
  proc(target, attachment, renderbuffertarget, renderbuffer);
}
void glRenderbufferStorageEXT(GLenum target, GLenum internalformat, GLsizei width, GLsizei height)
{
  static auto proc = reinterpret_cast<void(*)(GLenum, GLenum, GLsizei, GLsizei)>(_wglGetProcAddress(__func__));
  proc(target, internalformat, width, height);
}
} // anonymous namespace
#endif // defined(_WIN32) && !defined(GL_RENDERBUFFER_EXT)
#else
# define NO_OGL_ERR CV_Error(cv::Error::OpenGlNotSupported, \
                    "OpenGL support not enabled. Please rebuild the library with OpenGL support");
#endif

namespace cv {
namespace dynafu {
using namespace kinfu;

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

    Affine3f getPose() const CV_OVERRIDE;

    bool update(InputArray depth) CV_OVERRIDE;

    bool updateT(const T& depth);

    std::vector<Point3f> getNodesPos() const CV_OVERRIDE;

    void marchCubes(OutputArray vertices, OutputArray edges) const CV_OVERRIDE;

    void renderSurface(OutputArray depthImage, OutputArray vertImage, OutputArray normImage, bool warp=true) CV_OVERRIDE;

private:
    Params params;

    cv::Ptr<ICP> icp;
    cv::Ptr<NonRigidICP> dynafuICP;
    cv::Ptr<TSDFVolume> volume;

    int frameCounter;
    Affine3f pose;
    std::vector<T> pyrPoints;
    std::vector<T> pyrNormals;

    WarpField warpfield;

#ifdef HAVE_OPENGL
    ogl::Arrays arr;
    ogl::Buffer idx;
#endif
    void drawScene(OutputArray depthImg, OutputArray shadedImg);
};

template< typename T>
std::vector<Point3f> DynaFuImpl<T>::getNodesPos() const {
    NodeVectorType nv = warpfield.getNodes();
    std::vector<Point3f> nodesPos;
    for(auto n: nv)
        nodesPos.push_back(n->pos);

    return nodesPos;
}

template< typename T >
DynaFuImpl<T>::DynaFuImpl(const Params &_params) :
    params(_params),
    icp(makeICP(params.intr, params.icpIterations, params.icpAngleThresh, params.icpDistThresh)),
    dynafuICP(makeNonRigidICP(params.intr, volume, 2)),
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

    // Make a color attachment to this framebuffer
    unsigned int fbo_color;
    glGenRenderbuffersEXT(1, &fbo_color);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, fbo_color);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RGB, params.frameSize.width, params.frameSize.height);

    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_RENDERBUFFER_EXT, fbo_color);

#endif

    reset();
}

template< typename T >
void DynaFuImpl<T>::drawScene(OutputArray depthImage, OutputArray shadedImage)
{
#ifdef HAVE_OPENGL
    glViewport(0, 0, params.frameSize.width, params.frameSize.height);

    glEnable(GL_DEPTH_TEST);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

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

    Mat depthData(params.frameSize.height, params.frameSize.width, CV_32F);
    Mat shadeData(params.frameSize.height, params.frameSize.width, CV_32FC3);
    glReadPixels(0, 0, params.frameSize.width, params.frameSize.height, GL_DEPTH_COMPONENT, GL_FLOAT, depthData.ptr());
    glReadPixels(0, 0, params.frameSize.width, params.frameSize.height, GL_RGB, GL_FLOAT, shadeData.ptr());

    // linearise depth
    for(auto it = depthData.begin<float>(); it != depthData.end<float>(); ++it)
    {
        *it = farZ * nearZ / ((*it)*(nearZ - farZ) + farZ);

        if(*it >= farZ)
            *it = std::numeric_limits<float>::quiet_NaN();
    }

    if(depthImage.needed()) {
        depthData.copyTo(depthImage);
    }

    if(shadedImage.needed()) {
        shadeData.copyTo(shadedImage);
    }
#else
    CV_UNUSED(depthImage);
    CV_UNUSED(shadedImage);
    NO_OGL_ERR;
#endif
}

template< typename T >
void DynaFuImpl<T>::reset()
{
    frameCounter = 0;
    pose = Affine3f::Identity();
    warpfield.setAllRT(Affine3f::Identity());
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
Affine3f DynaFuImpl<T>::getPose() const
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
        warpfield.setAllRT(Affine3f::Identity());
    }
    else
    {
        UMat wfPoints;
        volume->fetchPointsNormals(wfPoints, noArray(), true);
        warpfield.updateNodesFromPoints(wfPoints);

        Mat _depthRender, estdDepth, _vertRender, _normRender;
        renderSurface(_depthRender, _vertRender, _normRender, false);
        _depthRender.convertTo(estdDepth, DEPTH_TYPE);

        std::vector<T> estdPoints, estdNormals;
        makeFrameFromDepth(estdDepth, estdPoints, estdNormals, params.intr,
                    params.pyramidLevels,
                    1.f,
                    params.bilateral_sigma_depth,
                    params.bilateral_sigma_spatial,
                    params.bilateral_kernel_size,
                    params.truncateThreshold);

        pyrPoints = estdPoints;
        pyrNormals = estdNormals;

        Affine3f affine;
        bool success = icp->estimateTransform(affine, pyrPoints, pyrNormals, newPoints, newNormals);
        if(!success)
            return false;

        pose = pose * affine;

        for(int iter = 0; iter < 1; iter++)
        {
            renderSurface(_depthRender, _vertRender, _normRender);
            _depthRender.convertTo(estdDepth, DEPTH_TYPE);

            makeFrameFromDepth(estdDepth, estdPoints, estdNormals, params.intr,
                params.pyramidLevels,
                1.f,
                params.bilateral_sigma_depth,
                params.bilateral_sigma_spatial,
                params.bilateral_kernel_size,
                params.truncateThreshold);

            success = dynafuICP->estimateWarpNodes(warpfield, pose, _vertRender, estdPoints[0],
                                                estdNormals[0],
                                               newPoints[0], newNormals[0]);
            if(!success)
                return false;
        }

        float rnorm = (float)cv::norm(affine.rvec());
        float tnorm = (float)cv::norm(affine.translation());
        // We do not integrate volume if camera does not move
        if((rnorm + tnorm)/2 >= params.tsdf_min_camera_movement)
        {
            // use depth instead of distance
            volume->integrate(depth, params.depthFactor, pose, params.intr, makePtr<WarpField>(warpfield));
        }


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
void DynaFuImpl<T>::renderSurface(OutputArray depthImage, OutputArray vertImage, OutputArray normImage, bool warp)
{
#ifdef HAVE_OPENGL
    Mat _vertices, vertices, normals, meshIdx;
    volume->marchCubes(_vertices, noArray());
    if(_vertices.empty()) return;

    _vertices.convertTo(vertices, POINT_TYPE);
    getNormals(vertices, normals);

    Mat warpedVerts(vertices.size(), vertices.type());

    Affine3f invCamPose(pose.inv());
    for(int i = 0; i < vertices.size().height; i++) {
        ptype v = vertices.at<ptype>(i);

        // transform vertex to RGB space
        Point3f pVoxel = (params.volumePose.inv() * Point3f(v[0], v[1], v[2])) / params.voxelSize;
        Point3f pGlobal = Point3f(pVoxel.x / params.volumeDims[0],
                                  pVoxel.y / params.volumeDims[1],
                                  pVoxel.z / params.volumeDims[2]);
        vertices.at<ptype>(i) = ptype(pGlobal.x, pGlobal.y, pGlobal.z, 1.f);

        // transform normals to RGB space
        ptype n = normals.at<ptype>(i);
        Point3f nGlobal = params.volumePose.rotation().inv() * Point3f(n[0], n[1], n[2]);
        nGlobal.x = (nGlobal.x + 1)/2;
        nGlobal.y = (nGlobal.y + 1)/2;
        nGlobal.z = (nGlobal.z + 1)/2;
        normals.at<ptype>(i) = ptype(nGlobal.x, nGlobal.y, nGlobal.z, 1.f);

        //Point3f p = Point3f(v[0], v[1], v[2]);

        if(!warp)
        {
            Point3f p(invCamPose * params.volumePose * (pVoxel*params.voxelSize));
            warpedVerts.at<ptype>(i) = ptype(p.x, p.y, p.z, 1.f);
        }
        else
        {
            int numNeighbours = 0;
            const nodeNeighboursType neighbours = volume->getVoxelNeighbours(pVoxel, numNeighbours);
            Point3f p = (invCamPose * params.volumePose) * warpfield.applyWarp(pVoxel*params.voxelSize, neighbours, numNeighbours);
            warpedVerts.at<ptype>(i) = ptype(p.x, p.y, p.z, 1.f);
        }
    }

    for(int i = 0; i < vertices.size().height; i++)
        meshIdx.push_back<int>(i);

    arr.setVertexArray(warpedVerts);
    arr.setColorArray(vertices);
    idx.copyFrom(meshIdx);

    drawScene(depthImage, vertImage);

    arr.setVertexArray(warpedVerts);
    arr.setColorArray(normals);
    drawScene(noArray(), normImage);
#else
    CV_UNUSED(depthImage);
    CV_UNUSED(vertImage);
    CV_UNUSED(normImage);
    CV_UNUSED(warp);
    NO_OGL_ERR;
#endif
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
