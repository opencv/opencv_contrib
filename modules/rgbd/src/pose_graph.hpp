#ifndef OPENCV_RGBD_GRAPH_NODE_H
#define OPENCV_RGBD_GRAPH_NODE_H

#include <map>
#include <unordered_map>

#include "opencv2/core/affine.hpp"
#if defined(HAVE_EIGEN)
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "opencv2/core/eigen.hpp"
#endif

#if defined(CERES_FOUND)
#include <ceres/ceres.h>
#endif

#include "sparse_block_matrix.hpp"

#include "opencv2/core/dualquaternion.hpp"

namespace cv
{
namespace kinfu
{
/*! \class GraphNode
 *  \brief Defines a node/variable that is optimizable in a posegraph
 *
 *  Detailed description
 */

struct Pose3d
{
    Vec3d t;
    Vec4d vq;

    Pose3d() : t(), vq(1, 0, 0, 0) { }

    Pose3d(const Matx33d& rotation, const Vec3d& translation)
        : t(translation)
    {
        vq = Quatd::createFromRotMat(rotation).normalize().toVec();
    }

    explicit Pose3d(const Matx44d& pose) :
        Pose3d(pose.get_minor<3, 3>(0, 0), Vec3d(pose(0, 3), pose(1, 3), pose(2, 3)))
    { }

    // NOTE: Eigen overloads quaternion multiplication appropriately
    inline Pose3d operator*(const Pose3d& otherPose) const
    {
        Pose3d out(*this);
        out.t += Quatd(vq).toRotMat3x3(QUAT_ASSUME_UNIT) * otherPose.t;
        out.vq = (Quatd(out.vq) * Quatd(otherPose.vq)).toVec();
        return out;
    }

    Affine3d getAffine() const
    {
        return Affine3d(Quatd(vq).toRotMat3x3(QUAT_ASSUME_UNIT), t);
    }

    inline Pose3d inverse() const
    {
        Pose3d out;
        out.vq = Quatd(vq).conjugate().toVec();
        out.t = - (Quatd(out.vq).toRotMat3x3(QUAT_ASSUME_UNIT) * t);
        return out;
    }

    inline void normalizeRotation()
    {
        vq = Quatd(vq).normalize().toVec();
    }
};

struct PoseGraphNode
{
   public:
    explicit PoseGraphNode(int _nodeId, const Affine3d& _pose)
        : nodeId(_nodeId), isFixed(false), pose(_pose)
    {
        se3Pose = Pose3d(_pose.rotation(), _pose.translation());
    }
    virtual ~PoseGraphNode() = default;

    int getId() const { return nodeId; }
    inline Affine3d getPose() const
    {
        pose = se3Pose.getAffine();
        return pose;
    }
    void setPose(const Affine3d& _pose)
    {
        pose = _pose;
        se3Pose = Pose3d(pose.rotation(), pose.translation());
    }
    void setPose(const Pose3d& _pose)
    {
        se3Pose = _pose;
        pose = se3Pose.getAffine();
    }
    void setFixed(bool val = true) { isFixed = val; }
    bool isPoseFixed() const { return isFixed; }

   public:
    int nodeId;
    bool isFixed;
    mutable Affine3d pose;
    Pose3d se3Pose;
};

/*! \class PoseGraphEdge
 *  \brief Defines the constraints between two PoseGraphNodes
 *
 *  Detailed description
 */
struct PoseGraphEdge
{
   public:
    PoseGraphEdge(int _sourceNodeId, int _targetNodeId, const Affine3f& _transformation,
                  const Matx66f& _information = Matx66f::eye())
        : sourceNodeId(_sourceNodeId),
          targetNodeId(_targetNodeId),
          transformation(_transformation),
          information(_information)
    {
    }
    virtual ~PoseGraphEdge() = default;

    int getSourceNodeId() const { return sourceNodeId; }
    int getTargetNodeId() const { return targetNodeId; }

    bool operator==(const PoseGraphEdge& edge)
    {
        if ((edge.getSourceNodeId() == sourceNodeId && edge.getTargetNodeId() == targetNodeId) ||
            (edge.getSourceNodeId() == targetNodeId && edge.getTargetNodeId() == sourceNodeId))
            return true;
        return false;
    }

   public:
    int sourceNodeId;
    int targetNodeId;
    Affine3f transformation;
    Matx66f information;
};

//! @brief Reference: A tutorial on SE(3) transformation parameterizations and on-manifold
//! optimization Jose Luis Blanco Compactly represents the jacobian of the SE3 generator
// clang-format off
/* static const std::array<Matx44f, 6> generatorJacobian = { */
/*     // alpha */
/*     Matx44f(0, 0,  0, 0, */
/*             0, 0, -1, 0, */
/*             0, 1,  0, 0, */
/*             0, 0,  0, 0), */
/*     // beta */
/*     Matx44f( 0, 0, 1, 0, */
/*              0, 0, 0, 0, */
/*             -1, 0, 0, 0, */
/*              0, 0, 0, 0), */
/*     // gamma */
/*     Matx44f(0, -1, 0, 0, */
/*             1,  0, 0, 0, */
/*             0,  0, 0, 0, */
/*             0,  0, 0, 0), */
/*     // x */
/*     Matx44f(0, 0, 0, 1, */
/*             0, 0, 0, 0, */
/*             0, 0, 0, 0, */
/*             0, 0, 0, 0), */
/*     // y */
/*     Matx44f(0, 0, 0, 0, */
/*             0, 0, 0, 1, */
/*             0, 0, 0, 0, */
/*             0, 0, 0, 0), */
/*     // z */
/*     Matx44f(0, 0, 0, 0, */
/*             0, 0, 0, 0, */
/*             0, 0, 0, 1, */
/*             0, 0, 0, 0) */
/* }; */
// clang-format on

class PoseGraph
{
   public:


    explicit PoseGraph() {};
    virtual ~PoseGraph() = default;

    //! PoseGraph can be copied/cloned
    PoseGraph(const PoseGraph&) = default;
    PoseGraph& operator=(const PoseGraph&) = default;

    // can be used for debugging
    PoseGraph(const std::string& g2oFileName);

    void addNode(const PoseGraphNode& node)
    {
        int id = node.getId();
        const auto& it = nodes.find(id);
        if (it != nodes.end())
        {
            std::cout << "duplicated node, id=" << id << std::endl;
            nodes.insert(it, { id, node });
        }
        else
        {
            nodes.insert({ id, node });
        }
    }
    void addEdge(const PoseGraphEdge& edge) { edges.push_back(edge); }

    bool nodeExists(int nodeId) const
    {
        return (nodes.find(nodeId) != nodes.end());
        /*
        return std::find_if(nodes.begin(), nodes.end(), [nodeId](const PoseGraphNode& currNode) {
                   return currNode.getId() == nodeId;
               }) != nodes.end();
        */
    }

    bool isValid() const;

    int getNumNodes() const { return int(nodes.size()); }
    int getNumEdges() const { return int(edges.size()); }

   public:

    std::map<int, PoseGraphNode> nodes;
    std::vector<PoseGraphEdge>   edges;
};

namespace Optimizer
{
void optimize(PoseGraph& poseGraph);

#if defined(CERES_FOUND)
void createOptimizationProblem(PoseGraph& poseGraph, ceres::Problem& problem);

//! Error Functor required for Ceres to obtain an auto differentiable cost function
class Pose3dErrorFunctor
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Pose3dErrorFunctor(const Pose3d& _poseMeasurement, const Matx66d& _sqrtInformation)
        : poseMeasurement(_poseMeasurement)
    {
        cv2eigen(_sqrtInformation, sqrtInfo);
    }
    Pose3dErrorFunctor(const Pose3d& _poseMeasurement,
                       const Eigen::Matrix<double, 6, 6>& _sqrtInformation)
        : poseMeasurement(_poseMeasurement), sqrtInfo(_sqrtInformation)
    {
    }

    template<typename T>
    bool operator()(const T* const _pSourceTrans, const T* const _pSourceQuat,
                    const T* const _pTargetTrans, const T* const _pTargetQuat, T* _pResidual) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> sourceTrans(_pSourceTrans);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> targetTrans(_pTargetTrans);

        //Eigen::Quaternion<T> sourceQuat(_pSourceQuat[3], _pSourceQuat[0], _pSourceQuat[1], _pSourceQuat[2]);
        //Eigen::Quaternion<T> targetQuat(_pTargetQuat[3], _pTargetQuat[0], _pTargetQuat[1], _pTargetQuat[2]);
        Eigen::Quaternion<T> sourceQuat(_pSourceQuat[0], _pSourceQuat[1], _pSourceQuat[2], _pSourceQuat[3]);
        Eigen::Quaternion<T> targetQuat(_pTargetQuat[0], _pTargetQuat[1], _pTargetQuat[2], _pTargetQuat[3]);
        
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(_pResidual);

        //Eigen::Quaternion<T> targetQuatInv = targetQuat.conjugate();

        //Eigen::Quaternion<T> relativeQuat    = targetQuatInv * sourceQuat;
        //Eigen::Matrix<T, 3, 1> relativeTrans = targetQuatInv * (targetTrans - sourceTrans);

        Eigen::Quaternion<T> sourceQuatInv = sourceQuat.conjugate();

        Eigen::Quaternion<T> relativeQuat = sourceQuatInv * targetQuat;
        Eigen::Matrix<T, 3, 1> relativeTrans = sourceQuatInv * (targetTrans - sourceTrans);

        //! Definition should actually be relativeQuat * poseMeasurement.r.conjugate()
        Eigen::Quaternion<T> pmr(T(poseMeasurement.vq[0]), T(poseMeasurement.vq[1]), T(poseMeasurement.vq[2]), T(poseMeasurement.vq[3]));
        
        //DEBUG
        //Eigen::Quaternion<T> deltaRot = pmr * relativeQuat.conjugate();
        Eigen::Quaternion<T> deltaRot = relativeQuat.conjugate() * pmr;

        //DEBUG
        //deltaRot = (deltaRot.w() < T(0)) ? deltaRot * Eigen::Quaternion<T>(T(-1), T(0), T(0), T(0)) : deltaRot;

        Eigen::Matrix<T, 3, 1> pmt(T(poseMeasurement.t[0]), T(poseMeasurement.t[1]), T(poseMeasurement.t[2]));
        residual.template block<3, 1>(0, 0) = relativeTrans -  pmt;
        residual.template block<3, 1>(3, 0) = T(2.0) * deltaRot.vec();

        residual.applyOnTheLeft(sqrtInfo.template cast<T>());

        return true;
    }

    static ceres::CostFunction* create(const Pose3d& _poseMeasurement,
                                       const Matx66f& _sqrtInformation)
    {
        return new ceres::AutoDiffCostFunction<Pose3dErrorFunctor, 6, 3, 4, 3, 4>(
            new Pose3dErrorFunctor(_poseMeasurement, _sqrtInformation));
    }

   private:
    const Pose3d poseMeasurement;
    Eigen::Matrix<double, 6, 6> sqrtInfo;
};

// matrix form of Im(a)
const Matx44d M_Im{ 0, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1 };

// matrix form of conjugation
const Matx44d M_Conj{ 1,  0,  0,  0,
                      0, -1,  0,  0,
                      0,  0, -1,  0,
                      0,  0,  0, -1 };

// matrix form of quaternion multiplication from left side
inline Matx44d m_left(Quatd q)
{
    // M_left(a)* V(b) =
    //    = (I_4 * a0 + [ 0 | -av    [    0 | 0_1x3
    //                   av | 0_3] +  0_3x1 | skew(av)]) * V(b)

    float w = q.w, x = q.x, y = q.y, z = q.z;
    return { w, -x, -y, -z,
             x,  w, -z,  y,
             y,  z,  w, -x,
             z, -y,  x,  w };
}

// matrix form of quaternion multiplication from right side
inline Matx44d m_right(Quatd q)
{
    // M_right(b)* V(a) =
    //    = (I_4 * b0 + [ 0 | -bv    [    0 | 0_1x3
    //                   bv | 0_3] +  0_3x1 | skew(-bv)]) * V(a)

    float w = q.w, x = q.x, y = q.y, z = q.z;
    return { w, -x, -y, -z,
             x,  w,  z, -y,
             y, -z,  w,  x,
             z,  y, -x,  w };
}

// concatenate matrices vertically
template<typename _Tp, int m, int n, int k> static inline
Matx<_Tp, m + k, n> concatVert(const Matx<_Tp, m, n>& a, const Matx<_Tp, k, n>& b)
{
    Matx<_Tp, m + k, n> res;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(i, j) = a(i, j);
        }
    }
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(m + i, j) = b(i, j);
        }
    }
    return res;
}

// concatenate matrices horizontally
template<typename _Tp, int m, int n, int k> static inline
Matx<_Tp, m, n + k> concatHor(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, k>& b)
{
    Matx<_Tp, m, n + k> res;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(i, j) = a(i, j);
        }
    }
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            res(i, n + j) = b(i, j);
        }
    }
    return res;
}


class Pose3dAnalyticCostFunction : public ceres::SizedCostFunction<6, 3, 4, 3, 4>
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Pose3dAnalyticCostFunction(Vec3d _transMeasured, Quatd _rotMeasured, Matx66d _infoMatrix) :
        transMeasured(_transMeasured), rotMeasured(_rotMeasured), infoMatrix(_infoMatrix),
        sqrtInfoMatrix(llt6(infoMatrix))
    { }

    static Matx66d llt6(Matx66d m)
    {
        Matx66d L;
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < (i + 1); j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                    sum += L(i, k) * L(j, k);

                if (i == j)
                    L(i, i) = sqrt(m(i, i) - sum);
                else
                    L(i, j) = (1.0 / L(j, j) * (m(i, j) - sum));
            }
        }
        return L;
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override
    {
        Vec3d sourceTrans(parameters[0]);
        Vec4d p1(parameters[1]);
        Quatd sourceQuat(p1);
        Vec3d targetTrans(parameters[2]);
        Vec4d p3(parameters[3]);
        Quatd targetQuat(p3);

        // err_r = 2*Im(conj(rel_r) * measure_r) = 2*Im(conj(target_r) * source_r * measure_r)
        // err_t = conj(source_r) * (target_t - source_t) * source_r - measure_t

        Vec6d res;

        Quatd sourceQuatInv = sourceQuat.conjugate();
        Vec3d deltaTrans = targetTrans - sourceTrans;

        Quatd relativeQuat = sourceQuatInv * targetQuat;
        Vec3d relativeTrans = sourceQuatInv.toRotMat3x3(cv::QUAT_ASSUME_UNIT) * deltaTrans;

        //! Definition should actually be relativeQuat * rotMeasured.conjugate()
        Quatd deltaRot = relativeQuat.conjugate() * rotMeasured;

        Vec3d terr = relativeTrans - transMeasured;
        Vec3d rerr = 2.0 * Vec3d(deltaRot.x, deltaRot.y, deltaRot.z);
        Vec6d rterr(terr[0], terr[1], terr[2], rerr[0], rerr[1], rerr[2]);

        res = sqrtInfoMatrix * rterr;

        if (jacobians)
        {
            Matx<double, 6, 3> stj, ttj;
            Matx<double, 6, 4> sqj, tqj;

            // d(err_r) = 2*Im(d(conj(target_r) * source_r * measure_r)) = < measure_r is constant > =
            // 2*Im((conj(d(target_r)) * source_r + conj(target_r) * d(source_r)) * measure_r)
            // d(target_r) == 0:
            //  # d(err_r) = 2*Im(conj(target_r) * d(source_r) * measure_r)
            //  # V(d(err_r)) = 2 * M_Im * M_right(measure_r) * M_left(conj(target_r)) * V(d(source_r))
            //  # d(err_r) / d(source_r) = 2 * M_Im * M_right(measure_r) * M_left(conj(target_r))
            Matx34d drdsq = 2.0 * (m_right(rotMeasured) * m_left(targetQuat.conjugate())).get_minor<3, 4>(1, 0);

            // d(source_r) == 0:
            //  # d(err_r) = 2*Im(conj(d(target_r)) * source_r * measure_r)
            //  # V(d(err_r)) = 2 * M_Im * M_right(source_r * measure_r) * M_Conj * V(d(target_r))
            //  # d(err_r) / d(target_r) = 2 * M_Im * M_right(source_r * measure_r) * M_Conj
            Matx34d drdtq = 2.0 * (m_right(sourceQuat * rotMeasured) * M_Conj).get_minor<3, 4>(1, 0);

            // d(err_t) = d(conj(source_r) * (target_t - source_t) * source_r) =
            // conj(source_r) * (d(target_t) - d(source_t)) * source_r +
            // conj(d(source_r)) * (target_t - source_t) * source_r +
            // conj(source_r) * (target_t - source_t) * d(source_r) =
            // <conj(a*b) == conj(b)*conj(a), conj(target_t - source_t) = - (target_t - source_t), 2 * Im(x) = (x - conj(x))>
            // conj(source_r) * (d(target_t) - d(source_t)) * source_r +
            // 2 * Im(conj(source_r) * (target_t - source_t) * d(source_r))
            // d(*_t) == 0:
            //  # d(err_t) = 2 * Im(conj(source_r) * (target_t - source_t) * d(source_r))
            //  # V(d(err_t)) = 2 * M_Im * M_left(conj(source_r) * (target_t - source_t)) * V(d(source_r))
            //  # d(err_t) / d(source_r) = 2 * M_Im * M_left(conj(source_r) * (target_t - source_t))
            Vec3d td = targetTrans - sourceTrans;
            Matx34d dtdsq = 2 * m_left(sourceQuatInv * Quatd(0, deltaTrans[0], deltaTrans[1], deltaTrans[2])).get_minor<3, 4>(1, 0);
            // deltaTrans is rotated by sourceQuatInv, so the jacobian is rot matrix of sourceQuatInv by +1 or -1
            Matx33d dtdtt = sourceQuatInv.toRotMat3x3(QUAT_ASSUME_UNIT);
            Matx33d dtdst = - dtdtt;

            Matx33d z;
            sqj = concatVert(dtdsq, drdsq);
            tqj = concatVert(Matx34d(), drdtq);
            stj = concatVert(dtdst, z);
            ttj = concatVert(dtdtt, z);

            stj = sqrtInfoMatrix * stj;
            ttj = sqrtInfoMatrix * ttj;
            sqj = sqrtInfoMatrix * sqj;
            tqj = sqrtInfoMatrix * tqj;

            // sourceTrans
            if (jacobians[0])
            {
                for (int i = 0; i < 6 * 3; i++)
                    jacobians[0][i] = stj.val[i];
            }
            
            // sourceQuat
            if (jacobians[1])
            {
                for (int i = 0; i < 6 * 4; i++)
                    jacobians[1][i] = sqj.val[i];
            }

            // targetTrans
            if (jacobians[2])
            {
                for (int i = 0; i < 6 * 3; i++)
                    jacobians[2][i] = ttj.val[i];
            }

            // targetQuat
            if (jacobians[3])
            {
                for (int i = 0; i < 6 * 4; i++)
                    jacobians[3][i] = tqj.val[i];
            }
        }

        for (int i = 0; i < 6; i++)
            residuals[i] = res[i];

        return true;
    }

    static ceres::CostFunction* create(Vec3d _transMeasured, Quatd _rotMeasured, Matx66d _infoMatrix)
    {
        return new Pose3dAnalyticCostFunction(_transMeasured, _rotMeasured, _infoMatrix);
    }

    Vec3d transMeasured;
    Quatd rotMeasured;
    Matx66d infoMatrix;
    Matx66d sqrtInfoMatrix;
};


#endif

}  // namespace Optimizer

}  // namespace kinfu
}  // namespace cv
#endif /* ifndef OPENCV_RGBD_GRAPH_NODE_H */
