#include "opencv2/face.hpp"
#include "face_utils.hpp"
#include "precomp.hpp"

using namespace cv;
using namespace face;

int BasicFaceRecognizer::getNumComponents() const
{
    return _num_components;
}

void BasicFaceRecognizer::setNumComponents(int val)
{
    _num_components = val;
}

double BasicFaceRecognizer::getThreshold() const
{
    return _threshold;
}

void BasicFaceRecognizer::setThreshold(double val)
{
    _threshold = val;
}

std::vector<cv::Mat> BasicFaceRecognizer::getProjections() const
{
    return _projections;
}

cv::Mat BasicFaceRecognizer::getLabels() const
{
    return _labels;
}

cv::Mat BasicFaceRecognizer::getEigenValues() const
{
    return _eigenvalues;
}

cv::Mat BasicFaceRecognizer::getEigenVectors() const
{
    return _eigenvectors;
}

cv::Mat BasicFaceRecognizer::getMean() const
{
    return _mean;
}

void BasicFaceRecognizer::read(const FileNode& fs)
{
    //read matrices
    double _t = 0;
    fs["threshold"] >> _t; // older versions might not have "threshold"
    if (_t !=0)
        _threshold = _t;    // be careful, not to overwrite DBL_MAX with 0 !
    fs["num_components"] >> _num_components;
    fs["mean"] >> _mean;
    fs["eigenvalues"] >> _eigenvalues;
    fs["eigenvectors"] >> _eigenvectors;
    // read sequences
    readFileNodeList(fs["projections"], _projections);
    fs["labels"] >> _labels;
    const FileNode& fn = fs["labelsInfo"];
    if (fn.type() == FileNode::SEQ)
    {
       _labelsInfo.clear();
       for (FileNodeIterator it = fn.begin(); it != fn.end();)
       {
           LabelInfo item;
           it >> item;
           _labelsInfo.insert(std::make_pair(item.label, item.value));
       }
    }
}

void BasicFaceRecognizer::write(FileStorage& fs) const
{
    // write matrices
    fs << "threshold" << _threshold;
    fs << "num_components" << _num_components;
    fs << "mean" << _mean;
    fs << "eigenvalues" << _eigenvalues;
    fs << "eigenvectors" << _eigenvectors;
    // write sequences
    writeFileNodeList(fs, "projections", _projections);
    fs << "labels" << _labels;
    fs << "labelsInfo" << "[";
    for (std::map<int, String>::const_iterator it = _labelsInfo.begin(); it != _labelsInfo.end(); it++)
        fs << LabelInfo(it->first, it->second);
    fs << "]";
}

bool BasicFaceRecognizer::empty() const
{
    return (_labels.empty());
}
