#ifndef __OPENCV_FACE_BASIC_HPP
#define __OPENCV_FACE_BASIC_HPP

#include "opencv2/face.hpp"
#include "precomp.hpp"

#include <set>
#include <limits>
#include <iostream>

using namespace cv;

// Reads a sequence from a FileNode::SEQ with type _Tp into a result vector.
template<typename _Tp>
inline void readFileNodeList(const FileNode& fn, std::vector<_Tp>& result) {
    if (fn.type() == FileNode::SEQ) {
        for (FileNodeIterator it = fn.begin(); it != fn.end();) {
            _Tp item;
            it >> item;
            result.push_back(item);
        }
    }
}

// Writes the a list of given items to a cv::FileStorage.
template<typename _Tp>
inline void writeFileNodeList(FileStorage& fs, const String& name,
                              const std::vector<_Tp>& items) {
    // typedefs
    typedef typename std::vector<_Tp>::const_iterator constVecIterator;
    // write the elements in item to fs
    fs << name << "[";
    for (constVecIterator it = items.begin(); it != items.end(); ++it) {
        fs << *it;
    }
    fs << "]";
}

inline Mat asRowMatrix(InputArrayOfArrays src, int rtype, double alpha=1, double beta=0) {
    // make sure the input data is a vector of matrices or vector of vector
    if(src.kind() != _InputArray::STD_VECTOR_MAT && src.kind() != _InputArray::STD_VECTOR_VECTOR) {
        String error_message = "The data is expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< std::vector<...> >).";
        CV_Error(Error::StsBadArg, error_message);
    }
    // number of samples
    size_t n = src.total();
    // return empty matrix if no matrices given
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src.getMat(0).total();
    // create data matrix
    Mat data((int)n, (int)d, rtype);
    // now copy data
    for(unsigned int i = 0; i < n; i++) {
        // make sure data can be reshaped, throw exception if not!
        if(src.getMat(i).total() != d) {
            String error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src.getMat(i).total());
            CV_Error(Error::StsBadArg, error_message);
        }
        // get a hold of the current row
        Mat xi = data.row(i);
        // make reshape happy by cloning for non-continuous matrices
        if(src.getMat(i).isContinuous()) {
            src.getMat(i).reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src.getMat(i).clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

// Utility structure to load/save face label info (a pair of int and string) via FileStorage
struct LabelInfo
{
    LabelInfo():label(-1), value("") {}
    LabelInfo(int _label, const String &_value): label(_label), value(_value) {}
    int label;
    String value;
    void write(cv::FileStorage& fs) const
    {
        fs << "{" << "label" << label << "value" << value << "}";
    }
    void read(const cv::FileNode& node)
    {
        label = (int)node["label"];
        value = (String)node["value"];
    }
    std::ostream& operator<<(std::ostream& out)
    {
        out << "{ label = " << label << ", " << "value = " << value.c_str() << "}";
        return out;
    }
};

inline void write(cv::FileStorage& fs, const String&, const LabelInfo& x)
{
    x.write(fs);
}

inline void read(const cv::FileNode& node, LabelInfo& x, const LabelInfo& default_value = LabelInfo())
{
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}


class BasicFaceRecognizerImpl : public cv::face::BasicFaceRecognizer
{
public:
    BasicFaceRecognizerImpl(int num_components = 0, double threshold = DBL_MAX)
        : _num_components(num_components), _threshold(threshold)
    {}

    void load(const FileStorage& fs)
    {
        //read matrices
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

    void save(FileStorage& fs) const
    {
        // write matrices
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

    CV_IMPL_PROPERTY(int, NumComponents, _num_components)
    CV_IMPL_PROPERTY(double, Threshold, _threshold)
    CV_IMPL_PROPERTY_RO(std::vector<cv::Mat>, Projections, _projections)
    CV_IMPL_PROPERTY_RO(cv::Mat, Labels, _labels)
    CV_IMPL_PROPERTY_RO(cv::Mat, EigenValues, _eigenvalues)
    CV_IMPL_PROPERTY_RO(cv::Mat, EigenVectors, _eigenvectors)
    CV_IMPL_PROPERTY_RO(cv::Mat, Mean, _mean)

protected:
    int _num_components;
    double _threshold;
    std::vector<Mat> _projections;
    Mat _labels;
    Mat _eigenvectors;
    Mat _eigenvalues;
    Mat _mean;
};

#endif // __OPENCV_FACE_BASIC_HPP

