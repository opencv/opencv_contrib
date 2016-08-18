#ifdef HAVE_OPENCV_DNN
typedef dnn::DictValue LayerId;
typedef std::vector<cv::dnn::Blob> vector_Blob;

template<>
bool pyopencv_to(PyObject *o, dnn::Blob &blob, const char *name);

template<> struct pyopencvVecConverter<dnn::Blob>
{
    static bool to(PyObject* obj, std::vector<dnn::Blob>& value, const ArgInfo info)
    {
        if (PyArray_Check(obj))
        {
            value.resize(1);
            return pyopencv_to(obj, value[0], info.name);
        }

        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<dnn::Blob>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<>
bool pyopencv_to(PyObject *o, std::vector<dnn::Blob> &blobs, const char *name) //required for Layer::blobs RW
{
    return pyopencvVecConverter<dnn::Blob>::to(o, blobs, ArgInfo(name, false));
}

template<>
bool pyopencv_to(PyObject *o, dnn::Blob &blob, const char *name)
{
    Mat &dst = blob.matRef();
    if (!pyopencv_to(o, dst, name))
        return false;

    if (PyArray_Check(o)) //try fix channels
    {
        PyArrayObject* oarr = (PyArrayObject*) o;

        if (PyArray_NDIM(oarr) == dst.dims)
            return true;

        int ndims = PyArray_NDIM(oarr);
        std::vector<int> shape(ndims);
        const npy_intp* _sizes = PyArray_DIMS(oarr);
        for (int i = 0; i < ndims; i++)
            shape[i] = (int)_sizes[i];

        dst = dst.reshape(1, ndims, &shape[0]);
    }

    return true;
}

template<>
PyObject *pyopencv_from(const dnn::Blob &blob)
{
    return pyopencv_from(blob.matRefConst());
}

template<>
bool pyopencv_to(PyObject *o, dnn::DictValue &dv, const char *name)
{
    (void)name;
    if (!o || o == Py_None)
        return true; //Current state will be used
    else if (PyLong_Check(o))
    {
        dv = dnn::DictValue((int64)PyLong_AsLongLong(o));
        return true;
    }
    else if (PyFloat_Check(o))
    {
        dv = dnn::DictValue(PyFloat_AS_DOUBLE(o));
        return true;
    }
    else if (PyString_Check(o))
    {
        dv = dnn::DictValue(String(PyString_AsString(o)));
        return true;
    }
    else
        return false;
}

template<>
bool pyopencv_to(PyObject *o, dnn::BlobShape &shape, const char *name)
{
    std::vector<int> data;
    if (!pyopencv_to_generic_vec(o, data, ArgInfo(name, false)))
        return false;

    shape = data.size() ? dnn::BlobShape((int)data.size(), &data[0]) : dnn::BlobShape::empty();
    return true;
}

template<>
PyObject *pyopencv_from(const dnn::BlobShape &shape)
{
    std::vector<int> data(shape.ptr(), shape.ptr() + shape.dims());
    return pyopencv_from_generic_vec(data);
}

#endif