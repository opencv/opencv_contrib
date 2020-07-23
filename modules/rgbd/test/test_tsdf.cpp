// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;

/** Reprojects screen point to camera space given z coord. */
struct Reprojector
{
    Reprojector() {}
    inline Reprojector(Matx33f intr)
    {
        fxinv = 1.f / intr(0, 0), fyinv = 1.f / intr(1, 1);
        cx = intr(0, 2), cy = intr(1, 2);
    }
    template<typename T>
    inline cv::Point3_<T> operator()(cv::Point3_<T> p) const
    {
        T x = p.z * (p.x - cx) * fxinv;
        T y = p.z * (p.y - cy) * fyinv;
        return cv::Point3_<T>(x, y, p.z);
    }

    float fxinv, fyinv, cx, cy;
};

template<class Scene>
struct RenderInvoker : ParallelLoopBody
{
    RenderInvoker(Mat_<float>& _frame, Affine3f _pose,
        Reprojector _reproj,
        float _depthFactor) : ParallelLoopBody(),
        frame(_frame),
        pose(_pose),
        reproj(_reproj),
        depthFactor(_depthFactor)
    { }

    virtual void operator ()(const cv::Range& r) const
    {
        for (int y = r.start; y < r.end; y++)
        {
            float* frameRow = frame[y];
            for (int x = 0; x < frame.cols; x++)
            {
                float pix = 0;

                Point3f orig = pose.translation();
                // direction through pixel
                Point3f screenVec = reproj(Point3f((float)x, (float)y, 1.f));
                float xyt = 1.f / (screenVec.x * screenVec.x +
                    screenVec.y * screenVec.y + 1.f);
                Point3f dir = normalize(Vec3f(pose.rotation() * screenVec));
                // screen space axis
                dir.y = -dir.y;

                const float maxDepth = 20.f;
                const float maxSteps = 256;
                float t = 0.f;
                for (int step = 0; step < maxSteps && t < maxDepth; step++)
                {
                    Point3f p = orig + dir * t;
                    float d = Scene::map(p);
                    if (d < 0.000001f)
                    {
                        float depth = std::sqrt(t * t * xyt);
                        pix = depth * depthFactor;
                        break;
                    }
                    t += d;
                }

                frameRow[x] = pix;
            }
        }
    }

    Mat_<float>& frame;
    Affine3f pose;
    Reprojector reproj;
    float depthFactor;
};

struct Scene
{
    virtual ~Scene() {}
    static Ptr<Scene> create(Size sz, Matx33f _intr, float _depthFactor);
    virtual Mat depth(Affine3f pose) = 0;
    virtual std::vector<Affine3f> getPoses() = 0;
};

struct SemisphereScene : Scene
{
    const int framesPerCycle = 72;
    const float nCycles = 0.25f;
    const Affine3f startPose = Affine3f(Vec3f(0.f, 0.f, 0.f), Vec3f(1.5f, 0.3f, -1.5f));

    Size frameSize;
    Matx33f intr;
    float depthFactor;
    static cv::Mat_<float> randTexture;

    SemisphereScene(Size sz, Matx33f _intr, float _depthFactor) :
        frameSize(sz), intr(_intr), depthFactor(_depthFactor)
    { }

    static float map(Point3f p)
    {
        float plane = p.y + 0.5f;

        Point3f boxPose = p - Point3f(-0.0f, 0.3f, 0.5f);
        float boxSize = 0.5f;
        float roundness = 0.08f;
        Point3f boxTmp;
        boxTmp.x = max(abs(boxPose.x) - boxSize, 0.0f);
        boxTmp.y = max(abs(boxPose.y) - boxSize, 0.0f);
        boxTmp.z = max(abs(boxPose.z) - boxSize, 0.0f);
        float roundBox = (float)cv::norm(boxTmp) - roundness;

        Point3f spherePose = p - Point3f(-0.0f, 0.3f, 0.0f);
        float sphereRadius = 0.5f;
        float sphere = (float)cv::norm(spherePose) - sphereRadius;
        float sphereMinusBox = max(sphere, -roundBox);

        float subSphereRadius = 0.05f;
        Point3f subSpherePose = p - Point3f(0.3f, -0.1f, -0.3f);
        float subSphere = (float)cv::norm(subSpherePose) - subSphereRadius;

        float res = min({plane, sphereMinusBox, subSphere});
        return res;
    }

    Mat depth(Affine3f pose) override
    {
        Mat_<float> frame(frameSize);
        Reprojector reproj(intr);

        Range range(0, frame.rows);
        parallel_for_(range, RenderInvoker<SemisphereScene>(frame, pose, reproj, depthFactor));

        return std::move(frame);
    }

    std::vector<Affine3f> getPoses() override
    {
        std::vector<Affine3f> poses;
        for (int i = 0; i < framesPerCycle * nCycles; i++)
        {
            float angle = (float)(CV_2PI * i / framesPerCycle);
            Affine3f pose;
            pose = pose.rotate(startPose.rotation());
            pose = pose.rotate(Vec3f(0.f, -1.f, 0.f) * angle);
            pose = pose.translate(Vec3f(startPose.translation()[0] * sin(angle),
                startPose.translation()[1],
                startPose.translation()[2] * cos(angle)));
            poses.push_back(pose);
        }

        return poses;
    }

};

Ptr<Scene> Scene::create(Size sz, Matx33f _intr, float _depthFactor)
{
    return makePtr<SemisphereScene>(sz, _intr, _depthFactor);
}

struct Operator {
    void operator ()(Vec4f& vector, const int* position) const
    {
        if ( !isnan(vector[0]) )
        {
            float length = vector[0] * vector[0] +
                           vector[1] * vector[1] +
                           vector[2] * vector[2];
            //cout << length;
            ASSERT_LT(abs(1-length), 0.0001f);
        }
    }
};

static const bool display = false;

void normal_test(bool isHashTSDF)
{
    Ptr<kinfu::Params> _params;
    if (isHashTSDF)
        _params = kinfu::Params::hashTSDFParams(true);
    else
        _params = kinfu::Params::coarseParams();

    Ptr<Scene> scene = Scene::create(_params->frameSize, _params->intr, _params->depthFactor);
    std::vector<Affine3f> poses = scene->getPoses();

    Mat depth = scene->depth(poses[0]);
    UMat _points, _normals;

    Ptr<kinfu::Volume> volume = kinfu::makeVolume(_params->volumeType, _params->voxelSize, _params->volumePose, 
                                _params->raycast_step_factor, _params->tsdf_trunc_dist, _params->tsdf_max_weight, 
                                _params->truncateThreshold, _params->volumeDims);

    volume->integrate(depth, _params->depthFactor, poses[0], _params->intr);
    volume->raycast(poses[0], _params->intr, _params->frameSize, _points, _normals);
    
    //volume->fetchPointsNormals(_points, _normals);
    //volume->fetchNormals(_points, _normals);
 
    AccessFlag af = ACCESS_READ;
    Mat normals = _normals.getMat(af);
    normals.forEach<Vec4f>(Operator());

    if (display)
    {
        imshow("depth", depth * (1.f / _params->depthFactor / 4.f));
        Mat points = _points.getMat(af);
        Mat image;
        //renderPointsNormals(points, normals, image, _params->lightPose);
        imshow("render", image);
        waitKey(30000);
    }

    UMat _newPoints, _newNormals;
    volume->raycast(poses[17], _params->intr, _params->frameSize, _newPoints, _newNormals);

    normals = _newNormals.getMat(af);
    normals.forEach<Vec4f>(Operator());

    if (display)
    {
        imshow("depth", depth * (1.f / _params->depthFactor / 4.f));
        Mat points = _newPoints.getMat(af);
        Mat image;
        //renderPointsNormals(points, normals, image, _params->lightPose);
        imshow("render", image);
        waitKey(30000);
    }
}

TEST(TSDF, raycast_normals)
{
    normal_test(false);
}

TEST(HashTSDF, raycast_normals)
{
    normal_test(true);
}

}}  // namespace
