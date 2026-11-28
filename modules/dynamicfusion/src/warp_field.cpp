#include <nanoflann/nanoflann.hpp>
#include <opencv2/dynamicfusion/utils/dual_quaternion.hpp>
#include <opencv2/dynamicfusion/utils/knn_point_cloud.hpp>
#include <opencv2/dynamicfusion/types.hpp>
#include <opencv2/dynamicfusion/warp_field.hpp>
#include <opencv2/dynamicfusion/cuda/internal.hpp>
#include <opencv2/dynamicfusion/cuda/precomp.hpp>
#include <opencv2/dynamicfusion/optimisation.hpp>

using namespace cv::kfusion;
using namespace cv;
std::vector<utils::DualQuaternion<float>> neighbours; //THIS SHOULD BE SOMEWHERE ELSE BUT TOO SLOW TO REINITIALISE
utils::PointCloud cloud;

WarpField::WarpField()
{
    nodes = new std::vector<deformation_node>();
    index = new kd_tree_t(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    ret_index = std::vector<size_t>(KNN_NEIGHBOURS);
    out_dist_sqr = std::vector<float>(KNN_NEIGHBOURS);
    resultSet = new nanoflann::KNNResultSet<float>(KNN_NEIGHBOURS);
    resultSet->init(&ret_index[0], &out_dist_sqr[0]);
    neighbours = std::vector<utils::DualQuaternion<float>>(KNN_NEIGHBOURS);

}

WarpField::~WarpField()
{
    delete[] nodes;
    delete resultSet;
    delete index;
}

/**
 *
 * @param first_frame
 * @param normals
 */
void WarpField::init(const cv::Mat& first_frame, const cv::Mat& normals)
{
//    CV_Assert(first_frame.rows == normals.rows);
//    CV_Assert(first_frame.cols == normals.cols);
    nodes->resize(first_frame.cols * first_frame.rows);
    auto voxel_size = kfusion::KinFuParams::default_params_dynamicfusion().volume_size[0] /
                      kfusion::KinFuParams::default_params_dynamicfusion().volume_dims[0];
    for(int i = 0; i < first_frame.rows; i++)
        for(int j = 0; j < first_frame.cols; j++)
        {
            auto point = first_frame.at<Point>(i,j);
            auto norm = normals.at<Normal>(i,j);
            if(!std::isnan(point.x))
            {
                utils::Quaternion<float> r(Vec3f(norm.x,norm.y,norm.z));
                if(std::isnan(r.w_) || std::isnan(r.x_) ||std::isnan(r.y_) ||std::isnan(r.z_))
                    continue;

                utils::Quaternion<float> t(0,point.x, point.y, point.z);
                nodes->at(i*first_frame.cols+j).transform = utils::DualQuaternion<float>(t, r);

                nodes->at(i*first_frame.cols+j).vertex = Vec3f(point.x,point.y,point.z);
                nodes->at(i*first_frame.cols+j).weight = voxel_size;
            }
        }
    buildKDTree();
}

/**
 *
 * @param first_frame
 * @param normals
 */
void WarpField::init(const std::vector<Vec3f>& first_frame, const std::vector<Vec3f>& normals)
{
    nodes->resize(first_frame.size());
    auto voxel_size = kfusion::KinFuParams::default_params_dynamicfusion().volume_size[0] /
                      kfusion::KinFuParams::default_params_dynamicfusion().volume_dims[0];
    for (int i = 0; i < first_frame.size(); i++)
    {
        auto point = first_frame[i];
        auto norm = normals[i];
        if (!std::isnan(point[0]))
        {
            utils::Quaternion<float> t(0.f, point[0], point[1], point[2]);
            utils::Quaternion<float> r(norm);
            nodes->at(i).transform = utils::DualQuaternion<float>(t,r);

            nodes->at(i).vertex = point;
            nodes->at(i).weight = voxel_size;
        }
    }
    buildKDTree();
}

/**
 * \brief
 * \param frame
 * \param normals
 * \param pose
 * \param tsdfVolume
 * \param edges
 */
void WarpField::energy(const cuda::Cloud &frame,
                       const cuda::Normals &normals,
                       const Affine3f &pose,
                       const cuda::TsdfVolume &tsdfVolume,
                       const std::vector<std::pair<utils::DualQuaternion<float>, utils::DualQuaternion<float>>> &edges
)
{
//CV_Assert(normals.cols()==frame.cols());
//CV_Assert(normals.rows()==frame.rows());
}

/**
 *
 * @param canonical_vertices
 * @param canonical_normals
 * @param live_vertices
 * @param live_normals
 * @return
 */
float WarpField::energy_data(const std::vector<Vec3f> &canonical_vertices,
                             const std::vector<Vec3f> &canonical_normals,
                             const std::vector<Vec3f> &live_vertices,
                             const std::vector<Vec3f> &live_normals
)
{

    ceres::Problem problem;
    int i = 0;

    auto parameters = new double[nodes->size() * 6];
    std::vector<cv::Vec3d> double_vertices;
    for(auto v : canonical_vertices)
    {
        if(std::isnan(v[0]))
            continue;
        ceres::CostFunction* cost_function = DynamicFusionDataEnergy::Create(live_vertices[i], Vec3f(1,0,0), v, Vec3f(1,0,0), this);
        problem.AddResidualBlock(cost_function,
                                 NULL /* squared loss */,
                                 parameters);
        i++;
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    for(int i = 0; i < nodes->size() * 6; i++)
    {
        std::cout<<parameters[i]<<" ";
        if((i+1) % 6 == 0)
            std::cout<<std::endl;
    }



    float weights[KNN_NEIGHBOURS];
    auto canonical2 = canonical_vertices;
    for(auto v : canonical_vertices)
    {
        utils::Quaternion<float> rotation(0,0,0,0);
        Vec3f translation(0,0,0);
        getWeightsAndUpdateKNN(v, weights);
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
        {
            auto block_position = ret_index[i];
            std::cout<<ret_index[i]<<" Weight:"<<weights[i]<<" ";
            utils::Quaternion<float> rotation1(Vec3f(parameters[block_position],
                                                     parameters[block_position+1],
                                                     parameters[block_position+2]));
            rotation = rotation + weights[i] * rotation1 * nodes->at(block_position).transform.getRotation();

            Vec3f translation1(parameters[block_position+3],
                               parameters[block_position+4],
                               parameters[block_position+5]);
            Vec3f t;
            nodes->at(block_position).transform.getTranslation(t[0],t[1],t[2]);
            translation += weights[i]*t + translation1;
        }
        rotation.rotate(v);
        v += translation;
        std::cout<<std::endl<<"Value of v:"<<v<<std::endl;
    }
    for(auto v : canonical_vertices)
    {
        utils::DualQuaternion<float> final_quat = DQB(v, parameters);
        final_quat.transform(v);
        std::cout<<"Value of v[ "<<i<<" ]:"<<v<<std::endl;
    }

    delete[] parameters;
    return 0;
}
/**
 * \brief
 * \param edges
 */
void WarpField::energy_reg(const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
        kfusion::utils::DualQuaternion<float>>> &edges)
{

}

/**
 * Tukey loss function as described in http://web.as.uky.edu/statistics/users/pbreheny/764-F11/notes/12-1.pdf
 * \param x
 * \param c
 * \return
 *
 * \note
 * The value c = 4.685 is usually used for this loss function, and
 * it provides an asymptotic efficiency 95% that of linear
 * regression for the normal distribution
 *
 * In the paper, a value of 0.01 is suggested for c
 */
float WarpField::tukeyPenalty(float x, float c) const
{
    return std::abs(x) <= c ? x * std::pow((1 - (x * x) / (c * c)), 2) : 0.0f;
}

/**
 * Huber penalty function, implemented as described in https://en.wikipedia.org/wiki/Huber_loss
 * In the paper, a value of 0.0001 is suggested for delta
 * \param a
 * \param delta
 * \return
 */
float WarpField::huberPenalty(float a, float delta) const
{
    return std::abs(a) <= delta ? a * a / 2 : delta * std::abs(a) - delta * delta / 2;
}

/**
 *
 * @param points
 * @param normals
 */
void WarpField::warp(std::vector<Vec3f>& points, std::vector<Vec3f>& normals) const
{
    int i = 0;
    for (auto& point : points)
    {
        if(std::isnan(point[0]) || std::isnan(normals[i][0]))
            continue;
        KNN(point);
        utils::DualQuaternion<float> dqb = DQB(point);
        dqb.transform(point);
        point = warp_to_live * point;

        dqb.transform(normals[i]);
        normals[i] = warp_to_live * normals[i];
        i++;
    }
}

/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
utils::DualQuaternion<float> WarpField::DQB(const Vec3f& vertex) const
{
    utils::DualQuaternion<float> quaternion_sum;
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
        quaternion_sum = quaternion_sum + weighting(out_dist_sqr[ret_index[i]], nodes->at(ret_index[i]).weight) *
                                          nodes->at(ret_index[i]).transform;

    auto norm = quaternion_sum.magnitude();

    return utils::DualQuaternion<float>(quaternion_sum.getRotation() / norm.first,
                                        quaternion_sum.getTranslation() / norm.second);
}


/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
utils::DualQuaternion<float> WarpField::DQB(const Vec3f& vertex, double epsilon[KNN_NEIGHBOURS * 6]) const
{
    if(epsilon == NULL)
    {
        std::cerr<<"Invalid pointer in DQB"<<std::endl;
        exit(-1);
    }
    utils::DualQuaternion<float> quaternion_sum;
    utils::DualQuaternion<float> eps;
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
    {
        // epsilon [0:2] is rotation [3:5] is translation
        eps.from_twist(epsilon[i*6],epsilon[i*6 + 1],epsilon[i*6 + 2],epsilon[i*6 + 3],epsilon[i*6 + 4],epsilon[i*6 + 5]);
        quaternion_sum = quaternion_sum + weighting(out_dist_sqr[ret_index[i]], nodes->at(ret_index[i]).weight) *
                                          nodes->at(ret_index[i]).transform * eps;
    }

    auto norm = quaternion_sum.magnitude();

    return utils::DualQuaternion<float>(quaternion_sum.getRotation() / norm.first,
                                        quaternion_sum.getTranslation() / norm.second);
}

/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
void WarpField::getWeightsAndUpdateKNN(const Vec3f& vertex, float weights[KNN_NEIGHBOURS])
{
    KNN(vertex);
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
        // epsilon [0:2] is rotation [3:5] is translation
        weights[i] = weighting(out_dist_sqr[i], nodes->at(ret_index[i]).weight);
}

/**
 * \brief
 * \param squared_dist
 * \param weight
 * \return
 */
float WarpField::weighting(float squared_dist, float weight) const
{
    return (float) exp(-squared_dist / (2 * weight * weight));
}

/**
 * \brief
 * \return
 */
void WarpField::KNN(Vec3f point) const
{
//    resultSet->init(&ret_index[0], &out_dist_sqr[0]);
    index->findNeighbors(*resultSet, point.val, nanoflann::SearchParams(10));
}

/**
 * \brief
 * \return
 */
const std::vector<deformation_node>* WarpField::getNodes() const
{
    return nodes;
}

/**
 * \brief
 * \return
 */
void WarpField::buildKDTree()
{
    //    Build kd-tree with current warp nodes.
    cloud.pts.resize(nodes->size());
    for(size_t i = 0; i < nodes->size(); i++)
        cloud.pts[i] = nodes->at(i).vertex;
    index->buildIndex();
}

const cv::Mat WarpField::getNodesAsMat() const
{
    cv::Mat matrix(1, nodes->size(), CV_32FC3);
    for(int i = 0; i < nodes->size(); i++)
        matrix.at<cv::Vec3f>(i) = nodes->at(i).vertex;
    return matrix;
}

/**
 * \brief
 */
void WarpField::clear()
{

}
void WarpField::setWarpToLive(const Affine3f &pose)
{
    warp_to_live = pose;
}