#ifdef HAVE_OPENCV_SURFACE_MATCHING
typedef std::vector<ppf_match_3d::Pose3DPtr> vector_Pose3DPtr;
typedef std::string string;

template<>
bool pyopencv_to(PyObject *o, ppf_match_3d::Pose3DPtr &pose, const char *name);

template<> struct pyopencvVecConverter<ppf_match_3d::Pose3DPtr>
{
    static bool to(PyObject* obj, std::vector<ppf_match_3d::Pose3DPtr>& value, const ArgInfo info)
    {
        if (PyArray_Check(obj))
        {
            value.resize(1);
            return pyopencv_to(obj, value[0], info.name);
        }

        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<ppf_match_3d::Pose3DPtr>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<>
bool pyopencv_to(PyObject *o, std::vector<ppf_match_3d::Pose3DPtr> &poses, const char *name)
{
    return pyopencvVecConverter<ppf_match_3d::Pose3DPtr>::to(o, poses, ArgInfo(name, false));
}

#endif