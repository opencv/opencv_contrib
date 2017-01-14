#ifdef HAVE_OPENCV_RGBD
typedef std::vector<linemod::Template> vector_Template;
typedef std::vector<linemod::Match> vector_Match;
typedef std::vector< Ptr<linemod::Modality> > vector_Ptr_Modality;

template<>
bool pyopencv_to(PyObject *o, linemod::Feature &feature, const char *name);
template<>
bool pyopencv_to(PyObject *o, linemod::Template &_template, const char *name);
template<>
bool pyopencv_to(PyObject *o, linemod::Match &match, const char *name);
template<>
bool pyopencv_to(PyObject *o, Ptr<linemod::Modality> &modality, const char *name);

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

template<> struct pyopencvVecConverter<linemod::Template>
{
	static bool to(PyObject* obj, std::vector<linemod::Template>& value, const ArgInfo info)
	{
		if (PyArray_Check(obj))
		{
			value.resize(1);
			return pyopencv_to(obj, value[0], info.name);
		}

		return pyopencv_to_generic_vec(obj, value, info);
	}

	static PyObject* from(const std::vector<linemod::Template>& value)
	{
		return pyopencv_from_generic_vec(value);
	}
};

template<> struct pyopencvVecConverter<linemod::Match>
{
	static bool to(PyObject* obj, std::vector<linemod::Match>& value, const ArgInfo info)
	{
		if (PyArray_Check(obj))
		{
			value.resize(1);
			return pyopencv_to(obj, value[0], info.name);
		}

		return pyopencv_to_generic_vec(obj, value, info);
	}

	static PyObject* from(const std::vector<linemod::Match>& value)
	{
		return pyopencv_from_generic_vec(value);
	}
};

template<> struct pyopencvVecConverter< Ptr<linemod::Modality> >
{
	static bool to(PyObject* obj, std::vector< Ptr<linemod::Modality> >& value, const ArgInfo info)
	{
		if (PyArray_Check(obj))
		{
			value.resize(1);
			return pyopencv_to(obj, value[0], info.name);
		}

		return pyopencv_to_generic_vec(obj, value, info);
	}

	static PyObject* from(const std::vector< Ptr<linemod::Modality> >& value)
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
bool pyopencv_to(PyObject *o, std::vector<linemod::Template> &templates, const char *name) //required for std::vector<Template> templates
{
	return pyopencvVecConverter<linemod::Template>::to(o, templates, ArgInfo(name, false));
}

template<>
bool pyopencv_to(PyObject *o, std::vector<linemod::Match> &matches, const char *name) //required for std::vector<Match> matches
{
	return pyopencvVecConverter<linemod::Match>::to(o, matches, ArgInfo(name, false));
}

template<>
bool pyopencv_to(PyObject *o, std::vector< Ptr<linemod::Modality> > &modalities, const char *name) //required for std::vector< Ptr<Modality> > modalities
{
	return pyopencvVecConverter< Ptr<linemod::Modality> >::to(o, modalities, ArgInfo(name, false));
}

template<>
bool pyopencv_to(PyObject *o, linemod::Feature &feature, const char *name)
{
	(void)name;
	if (!o || o == Py_None)
		return true;
	if (PyObject_Size(o) == 0)
	{
		feature = linemod::Feature();
		return true;
	}
	return PyArg_ParseTuple(o, "iii", &feature.x, &feature.y, &feature.label) > 0;
}

template<>
PyObject *pyopencv_from(const linemod::Feature &feature)
{
	return Py_BuildValue("(iii)", feature.x, feature.y, feature.label);
}

template<>
bool pyopencv_to(PyObject *o, linemod::Template &_template, const char *name)
{
	(void)name;
	if (!o || o == Py_None)
		return true;
	if (PyObject_Size(o) == 0)
	{
		_template = linemod::Template();
		return true;
	}
	PyObject o_features;
	if (PyArg_ParseTuple(o, "iiiO", &_template.width, &_template.height, &_template.pyramid_level, &o_features) <= 0)
		return false;
	return pyopencv_to(&o_features, _template.features, name);
}

template<>
PyObject *pyopencv_from(const linemod::Template &_template)
{
	return Py_BuildValue("(iiiO)", _template.width, _template.height, _template.pyramid_level, pyopencv_from(_template.features));
}

template<>
bool pyopencv_to(PyObject *o, linemod::Match &match, const char *name)
{
	(void)name;
	if (!o || o == Py_None)
		return true;
	if (PyObject_Size(o) == 0)
	{
		match = linemod::Match();
		return true;
	}
	char class_id[128];
	if (PyArg_ParseTuple(o, "iifsi", &match.x, &match.y, &match.similarity, class_id, &match.template_id) <= 0)
		return false;
	match.class_id = String(class_id);
	return true;
}

template<>
PyObject *pyopencv_from(const linemod::Match &match)
{
	return Py_BuildValue("(iifsi)", match.x, match.y, match.similarity, match.class_id.c_str(), match.template_id);
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