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
    static Ptr<Scene> create(int nScene, Size sz, Matx33f _intr, float _depthFactor);
    virtual Mat depth(Affine3f pose) = 0;
    virtual std::vector<Affine3f> getPoses() = 0;
};

struct SemisphereScene : Scene
{
    const int framesPerCycle = 72;
    const float nCycles = 0.5f;
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

        return sphereMinusBox;
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

Ptr<Scene> Scene::create(int nScene, Size sz, Matx33f _intr, float _depthFactor)
{
    return makePtr<SemisphereScene>(sz, _intr, _depthFactor);
}

struct Operator {
    void operator ()(Vec4f& vector, const int* position) const
    {
        if ( !isnan(vector[0]) )
        {
            float tmp = vector[0] * vector[0] +
                        vector[1] * vector[1] +
                        vector[2] * vector[2];
            cout << tmp << endl;
        }
    }
};

static const bool display = true;

void nolmal_test(bool hiDense)
{
    Ptr<kinfu::Params> _params;
    _params = kinfu::Params::TSDFParams(hiDense);
    
    Ptr<Scene> scene = Scene::create(hiDense, _params->frameSize, _params->intr, _params->depthFactor);
    Ptr<kinfu::KinFu> kf = kinfu::KinFu::create(_params);
    std::vector<Affine3f> poses = scene->getPoses();

    for (size_t i = 0; i < poses.size(); i++)
    {
        Mat depth = scene->depth(poses[i]);
        kf->update(depth);

        //UMat points, normals;
        //kf->getPoints(points);
        //kf->getNormals(points, normals);

        //AccessFlag af = ACCESS_READ;
        //Mat norm = normals.getMat(af);
        //norm.forEach<Vec4f>(Operator());

        if (display)
        {
            imshow("depth", depth * (1.f / _params->depthFactor / 4.f));
            Mat rendered;
            kf->render(rendered);
            imshow("render", rendered);
            waitKey(500);
        }
    }
}

TEST(TSDF, normals_lowDense)
{
    nolmal_test(false);
}

TEST(TSDF, normals_highDense)
{
    nolmal_test(true);
}


}}  // namespace