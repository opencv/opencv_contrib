#ifdef HAVE_OPENCV_QDS
#include "opencv2/core/saturate.hpp"

template<> struct pyopencvVecConverter<qds::Match>
{
    static bool to(PyObject* obj, std::vector<qds::Match>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<qds::Match>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};
typedef std::vector<qds::Match> vector_qds_Match;

#endif
