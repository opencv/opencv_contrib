#include "opencv2/line_descriptor.hpp"

template<> struct pyopencvVecConverter<line_descriptor::KeyLine>
{
    static bool to(PyObject* obj, std::vector<line_descriptor::KeyLine>& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<line_descriptor::KeyLine>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

typedef std::vector<line_descriptor::KeyLine> vector_KeyLine;
typedef std::vector<std::vector<line_descriptor::KeyLine> > vector_vector_KeyLine;
