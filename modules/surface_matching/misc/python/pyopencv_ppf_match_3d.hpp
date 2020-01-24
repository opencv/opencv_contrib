#ifdef HAVE_OPENCV_SURFACE_MATCHING

template<> struct pyopencvVecConverter<ppf_match_3d::Pose3DPtr >
{
    static bool to(PyObject* obj, std::vector<ppf_match_3d::Pose3DPtr >& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<ppf_match_3d::Pose3DPtr >& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

typedef std::vector<ppf_match_3d::Pose3DPtr> vector_Pose3DPtr;
#endif
