#ifdef HAVE_OPENCV_DNN
typedef dnn::DictValue LayerId;

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

#endif
