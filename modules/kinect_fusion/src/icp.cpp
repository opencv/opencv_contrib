//TODO: add license

#include "precomp.hpp"

using namespace cv::kinfu;

//TODO: ICP class impl here
ICP::ICP(int nLevels) : levels(nLevels)
{
    //TODO: init somewhat
}

bool ICP::estimateTransform(cv::Affine3f& transform,
                            const std::vector<Points>& oldPoints, const std::vector<Normals>& oldNormals,
                            const std::vector<Points>& newPoints, const std::vector<Normals>& newNormals)
{
    //TODO:  implement this
    transform = cv::Affine3f().translate(cv::Vec3f(0.1f, 0.f, 0.f));
    return true;
}

