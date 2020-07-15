// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <unordered_map>

#include "opencv2/core/types.hpp"

#if defined(HAVE_EIGEN)
#include "opencv2/core/eigen.hpp"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#endif

namespace cv
{
namespace kinfu
{
/*!
 * \class BlockSparseMat
 * Naive implementation of Sparse Block Matrix
 */
template<typename _Tp, int blockM, int blockN>
struct BlockSparseMat
{
    typedef std::unordered_map<Point2i, Matx<_Tp, blockM, blockN>> IDtoBlockValueMap;
    static constexpr int blockSize = blockM * blockN;
    BlockSparseMat(int _nBlocks) : nBlocks(_nBlocks), ijValue() {}

    Matx66f& refBlock(int i, int j)
    {
        Point2i p(i, j);
        auto it = ijValue.find(p);
        if (it == ijValue.end())
        {
            it = ijValue.insert({ p, Matx<_Tp, blockM, blockN>() }).first;
        }
        return it->second;
    }

    float& refElem(int i, int j)
    {
        Point2i ib(i / blockSize, j / blockSize), iv(i % blockSize, j % blockSize);
        return refBlock(ib.x, ib.y)(iv.x, iv.y);
    }

    size_t nonZeroBlocks() const { return ijValue.size(); }

    int nBlocks;
    IDtoBlockValueMap ijValue;
};

//! Function to solve a sparse linear system of equations HX = B
//! Requires Eigen
static bool sparseSolve(const BlockSparseMat<float, 6, 6>& H, const Mat& B, Mat& X)
{
    const float matValThreshold = 0.001f;
    bool result = false;

#if defined(HAVE_EIGEN)
    std::cout << "starting eigen-insertion..." << std::endl;

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(H.ijValue.size() * H.blockSize * H.blockSize);
    for (auto ijValue : H.ijValue)
    {
        int xb = ijValue.first.x, yb = ijValue.first.y;
        Matx66f vblock = ijValue.second;
        for (int i = 0; i < H.blockSize; i++)
        {
            for (int j = 0; j < H.blockSize; j++)
            {
                float val = vblock(i, j);
                if (abs(val) >= matValThreshold)
                {
                    tripletList.push_back(Eigen::Triplet<double>(H.blockSize * xb + i, H.blockSize * yb + j, val));
                }
            }
        }
    }

    Eigen::SparseMatrix<float> bigA(H.blockSize * H.nBlocks, H.blockSize * H.nBlocks);
    bigA.setFromTriplets(tripletList.begin(), tripletList.end());
    bigA.makeCompressed();

    Eigen::VectorXf bigB;
    cv2eigen(B, bigB);

    //!TODO: Check if this is required
    if (!bigA.isApprox(bigA.transpose())
    {
        CV_Error(Error::StsBadArg, "Sparse Matrix is not symmetric");
        return result;
    }


    // TODO: try this, LLT and Cholesky
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>, Eigen::NaturalOrdering<int>> solver;

    std::cout << "starting eigen-compute..." << std::endl;
    solver.compute(bigA);

    if (solver.info() != Eigen::Success)
    {
        std::cout << "failed to eigen-decompose" << std::endl;
        result = false;
    }
    else
    {
        std::cout << "starting eigen-solve..." << std::endl;

        Eigen::VectorXf sx = solver.solve(bigB);
        if (solver.info() != Eigen::Success)
        {
            std::cout << "failed to eigen-solve" << std::endl;
            result = false;
        }
        else
        {
            x.resize(jtb.size);
            eigen2cv(sx, x);
            result = true;
        }
    }

#else
    std::cout << "no eigen library" << std::endl;

    CV_Error(Error::StsNotImplemented, "Eigen library required for matrix solve, dense solver is not implemented");
#endif

    return result;
}

}  // namespace kinfu
}  // namespace cv
