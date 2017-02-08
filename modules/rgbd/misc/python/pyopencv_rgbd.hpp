#ifdef HAVE_OPENCV_RGBD

template<>
bool pyopencv_to(PyObject *o, linemod::Feature &feature, const char *name);

template<> struct pyopencvVecConverter<linemod::Feature>
{
    static bool to(PyObject* obj, std::vector<linemod::Feature>& value, const ArgInfo info)
    {
        if (PyArray_Check(obj))
        {
            value.resize(1);
            return pyopencv_to(obj, value[0], info.name);
        }

        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<linemod::Feature>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<>
bool pyopencv_to(PyObject *o, std::vector<linemod::Feature> &features, const char *name) //required for std::vector<Feature> features
{
    return pyopencvVecConverter<linemod::Feature>::to(o, features, ArgInfo(name, false));
}

template<>
bool pyopencv_to(PyObject *o, linemod::Feature &feature, const char *name)
{
    std::vector<int> data;
    if (!pyopencv_to_generic_vec(o, data, ArgInfo(name, false)))
        return false;

    feature = data.size() ? linemod::Feature(data[0], data[1], data[2]) : linemod::Feature();
    return true;
}

template<>
PyObject *pyopencv_from(const linemod::Feature &feature)
{
    std::vector<int> data;
    data.push_back(feature.x);
    data.push_back(feature.y);
    data.push_back(feature.label);
    return pyopencv_from_generic_vec(data);
}

template<>
bool pyopencv_to(PyObject *o, std::vector<Mat> &mats, const char *name)
{
    return pyopencvVecConverter<Mat>::to(o, mats, ArgInfo(name, false));
}

template<>
bool pyopencv_to(PyObject *o, ushort &depth, const char *name)  //required for isValidDepth(const ushort & depth)
{
	std::vector<int> data;
	if (!pyopencv_to_generic_vec(o, data, ArgInfo(name, false)))
		return false;

	depth = data.size() ? ushort(data[0]) : 0;
	return true;
}

template<>
bool pyopencv_to(PyObject *o, short &depth, const char *name)  //required for isValidDepth(const short & depth)
{
	std::vector<int> data;
	if (!pyopencv_to_generic_vec(o, data, ArgInfo(name, false)))
		return false;

	depth = data.size() ? short(data[0]) : 0;
	return true;
}

template<>
bool pyopencv_to(PyObject *o, uint &depth, const char *name)  //required for isValidDepth(const uint & depth)
{
	std::vector<int> data;
	if (!pyopencv_to_generic_vec(o, data, ArgInfo(name, false)))
		return false;

	depth = data.size() ? uint(data[0]) : 0;
	return true;
}

#endif