#include <opencv2/dynamicfusion/cuda/precomp.hpp>


using namespace cv::kfusion;
using namespace std;
using namespace cv::kfusion::cuda;
using namespace cv;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ComputeIcpHelper

kfusion::device::ComputeIcpHelper::ComputeIcpHelper(float dist_thres, float angle_thres)
{
    min_cosine = cos(angle_thres);
    dist2_thres = dist_thres * dist_thres;
}

void kfusion::device::ComputeIcpHelper::setLevelIntr(int level_index, float fx, float fy, float cx, float cy)
{
    int div = 1 << level_index;
    f = make_float2(fx/div, fy/div);
    c = make_float2(cx/div, cy/div);
    finv = make_float2(1.f/f.x, 1.f/f.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ProjectiveICP::StreamHelper

struct kfusion::cuda::ProjectiveICP::StreamHelper
{
    typedef device::ComputeIcpHelper::PageLockHelper PageLockHelper;
    typedef cv::Matx66f Mat6f;
    typedef cv::Vec6f Vec6f;

    cudaStream_t stream;
    PageLockHelper locked_buffer;

    StreamHelper() { cudaSafeCall( cudaStreamCreate(&stream) ); }
    ~StreamHelper() { cudaSafeCall( cudaStreamDestroy(stream) ); }

    operator float*() { return locked_buffer.data; }
    operator cudaStream_t() { return stream; }

    Mat6f get(Vec6f& b)
    {
        cudaSafeCall( cudaStreamSynchronize(stream) );

        Mat6f A;
        float *data_A = A.val;
        float *data_b = b.val;

        int shift = 0;
        for (int i = 0; i < 6; ++i)   //rows
            for (int j = i; j < 7; ++j) // cols + b
            {
                float value = locked_buffer.data[shift++];
                if (j == 6)               // vector b
                    data_b[i] = value;
                else
                    data_A[j * 6 + i] = data_A[i * 6 + j] = value;
            }
        return A;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ProjectiveICP

kfusion::cuda::ProjectiveICP::ProjectiveICP() : angle_thres_(deg2rad(20.f)), dist_thres_(0.1f)
{ 
    const int iters[] = {10, 5, 4, 0};
    std::vector<int> vector_iters(iters, iters + 4);
    setIterationsNum(vector_iters);
    device::ComputeIcpHelper::allocate_buffer(buffer_);

    shelp_ = cv::Ptr<StreamHelper>(new StreamHelper());
}

kfusion::cuda::ProjectiveICP::~ProjectiveICP() {}

float kfusion::cuda::ProjectiveICP::getDistThreshold() const
{ return dist_thres_; }

void kfusion::cuda::ProjectiveICP::setDistThreshold(float distance)
{ dist_thres_ = distance; }

float kfusion::cuda::ProjectiveICP::getAngleThreshold() const
{ return angle_thres_; }

void kfusion::cuda::ProjectiveICP::setAngleThreshold(float angle)
{ angle_thres_ = angle; }

void kfusion::cuda::ProjectiveICP::setIterationsNum(const std::vector<int>& iters)
{
    if (iters.size() >= MAX_PYRAMID_LEVELS)
        iters_.assign(iters.begin(), iters.begin() + MAX_PYRAMID_LEVELS);
    else
    {
        iters_ = vector<int>(MAX_PYRAMID_LEVELS, 0);
        copy(iters.begin(), iters.end(),iters_.begin());
    }
}

int kfusion::cuda::ProjectiveICP::getUsedLevelsNum() const
{
    int i = MAX_PYRAMID_LEVELS - 1;
    for(; i >= 0 && !iters_[i]; --i);
    return i + 1;
}

bool kfusion::cuda::ProjectiveICP::estimateTransform(Affine3f& /*affine*/, const Intr& /*intr*/, const Frame& /*curr*/, const Frame& /*prev*/)
{
//    bool has_depth = !curr.depth_pyr.empty() && !prev.depth_pyr.empty();
//    bool has_points = !curr.points_pyr.empty() && !prev.points_pyr.empty();

//    if (has_depth)
//        return estimateTransform(affine, intr, curr.depth_pyr, curr.normals_pyr, prev.depth_pyr, prev.normals_pyr);
//    else if(has_points)
//         return estimateTransform(affine, intr, curr.points_pyr, curr.normals_pyr, prev.points_pyr, prev.normals_pyr);
//    else
//        CV_Assert(!"Wrong parameters passed to estimateTransform");
//    CV_Assert(!"Not implemented");
    return false;
}

bool kfusion::cuda::ProjectiveICP::estimateTransform(Affine3f& affine, const Intr& intr, const DepthPyr& dcurr, const NormalsPyr ncurr, const DepthPyr dprev, const NormalsPyr nprev)
{
    const int LEVELS = getUsedLevelsNum();
    StreamHelper& sh = *shelp_;

    device::ComputeIcpHelper helper(dist_thres_, angle_thres_);
    affine = Affine3f::Identity();

    for(int level_index = LEVELS - 1; level_index >= 0; --level_index)
    {
        const device::Normals& n = (const device::Normals& )nprev[level_index];

        helper.rows = (float)n.rows();
        helper.cols = (float)n.cols();
        helper.setLevelIntr(level_index, intr.fx, intr.fy, intr.cx, intr.cy);
        helper.dcurr = dcurr[level_index];
        helper.ncurr = ncurr[level_index];

        for(int iter = 0; iter < iters_[level_index]; ++iter)
        {
            helper.aff = device_cast<device::Aff3f>(affine);
            helper(dprev[level_index], n, buffer_, sh, sh);

            StreamHelper::Vec6f b;
            StreamHelper::Mat6f A  = sh.get(b);

            //checking nullspace
            double det = cv::determinant(A);

            if (fabs (det) < 1e-15 || cv::viz::isNan(det))
            {
                if (cv::viz::isNan(det)) cout << "qnan" << endl;
                return false;
            }

            StreamHelper::Vec6f r;
            cv::solve(A, b, r, cv::DECOMP_SVD);
            Affine3f Tinc(Vec3f(r.val), Vec3f(r.val+3));
            affine = Tinc * affine;
        }
    }
    return true;
}

bool kfusion::cuda::ProjectiveICP::estimateTransform(Affine3f& affine, const Intr& intr, const PointsPyr& vcurr, const NormalsPyr ncurr, const PointsPyr vprev, const NormalsPyr nprev)
{
    const int LEVELS = getUsedLevelsNum();
    StreamHelper& sh = *shelp_;

    device::ComputeIcpHelper helper(dist_thres_, angle_thres_);
    affine = Affine3f::Identity();

    for(int level_index = LEVELS - 1; level_index >= 0; --level_index)
    {
        const device::Normals& n = (const device::Normals& )nprev[level_index];
        const device::Points& v = (const device::Points& )vprev[level_index];

        helper.rows = (float)n.rows();
        helper.cols = (float)n.cols();
        helper.setLevelIntr(level_index, intr.fx, intr.fy, intr.cx, intr.cy);
        helper.vcurr = vcurr[level_index];
        helper.ncurr = ncurr[level_index];

        for(int iter = 0; iter < iters_[level_index]; ++iter)
        {
            helper.aff = device_cast<device::Aff3f>(affine);
            helper(v, n, buffer_, sh, sh);

            StreamHelper::Vec6f b;
            StreamHelper::Mat6f A = sh.get(b);

            //checking nullspace
            double det = cv::determinant(A);

            if (fabs (det) < 1e-15 || cv::viz::isNan (det))
            {
                if (cv::viz::isNan (det)) cout << "qnan" << endl;
                return false;
            }

            StreamHelper::Vec6f r;
            cv::solve(A, b, r, cv::DECOMP_SVD);

            Affine3f Tinc(Vec3f(r.val), Vec3f(r.val+3));
            affine = Tinc * affine;
        }
    }
    return true;
}

