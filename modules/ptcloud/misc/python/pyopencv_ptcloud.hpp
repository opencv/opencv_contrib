#include "opencv2/ptcloud/sac_segmentation.hpp"

// SACScore
template<>
PyObject* pyopencv_from(const std::pair<double, double>& src)
{
    return Py_BuildValue("(dd)", src.first, src.second);
}

// we have a vector of structs in c++, but python needs a list of Ptrs
// (in the end, this makes dreadful copies of anything)
template<>
PyObject* pyopencv_from(const std::vector<cv::ptcloud::SACModel>& src)
{
    int i, n = (int)src.size();
    PyObject* seq = PyList_New(n);
    for( i = 0; i < n; i++ )
    {
        Ptr<ptcloud::SACModel> ptr(new ptcloud::SACModel());
        *ptr = src[i];
        PyObject* item = pyopencv_from(ptr);
        if(!item)
            break;
        PyList_SetItem(seq, i, item);
    }
    if( i < n )
    {
        Py_DECREF(seq);
        return 0;
    }
    return seq;
}

typedef std::vector<ptcloud::SACModel> vector_SACModel;
