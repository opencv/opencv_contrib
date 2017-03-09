#ifdef HAVE_OPENCV_SURFACE_MATCHING
typedef std::vector<ppf_match_3d::Pose3DPtr> vector_Pose3DPtr;

template<>
bool pyopencv_to(PyObject *obj, ppf_match_3d::Pose3DPtr &pose3D, const char *name);

template<>
PyObject *pyopencv_from(const ppf_match_3d::Pose3DPtr &pose3D);

template<>
bool pyopencv_to(PyObject *obj, ppf_match_3d::Pose3D &pose3D, const char *name);

template<>
PyObject *pyopencv_from(const ppf_match_3d::Pose3D &pose3D);

template<> struct pyopencvVecConverter<ppf_match_3d::Pose3DPtr>
{
    static bool to(PyObject* obj, vector_Pose3DPtr& value, const ArgInfo info)
    {
        if (PyArray_Check(obj))
        {
            value.resize(1);
            return pyopencv_to(obj, value[0], info.name);
        }

        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const vector_Pose3DPtr& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<>
bool pyopencv_to(PyObject *obj, vector_Pose3DPtr &vector_pose3D, const char *name)
{
    return pyopencvVecConverter<ppf_match_3d::Pose3DPtr>::to(obj, vector_pose3D, ArgInfo(name, false));
}

template<>
bool pyopencv_to(PyObject *obj, ppf_match_3d::Pose3DPtr &pose3D, const char *name)
{
    if (pose3D.empty())
        pose3D.reset(new ppf_match_3d::Pose3D());
    return  pyopencv_to(obj, *pose3D, name);
}

template<>
PyObject *pyopencv_from(const ppf_match_3d::Pose3DPtr &pose3D) // required for PPF3DDetector::match
{
    return pyopencv_from(*pose3D);
}

template<>
bool pyopencv_to(PyObject *obj, ppf_match_3d::Pose3D &pose3D, const char *name)
{
    PyObject* attr = NULL;
    Matx44d pose;
    bool ok = false;

    (void)name;
    if (PyMapping_HasKeyString(obj, (char*)"modelIndex"))
    {
        attr = PyMapping_GetItemString(obj, (char*)"modelIndex");
        ok = attr && pyopencv_to(attr, pose3D.modelIndex);
        Py_DECREF(attr);
        if (!ok) return false;
    }
    if (PyMapping_HasKeyString(obj, (char*)"numVotes"))
    {
        attr = PyMapping_GetItemString(obj, (char*)"numVotes");
        ok = attr && pyopencv_to(attr, pose3D.numVotes);
        Py_DECREF(attr);
        if (!ok) return false;
    }
    if (PyMapping_HasKeyString(obj, (char*)"residual"))
    {
        attr = PyMapping_GetItemString(obj, (char*)"residual");
        ok = attr && pyopencv_to(attr, pose3D.residual);
        Py_DECREF(attr);
        if (!ok) return false;
    }
    if (PyMapping_HasKeyString(obj, (char*)"pose"))
    {
        attr = PyMapping_GetItemString(obj, (char*)"pose");
        ok = attr && pyopencv_to(attr, pose);
        Py_DECREF(attr);
        pose3D.updatePose(pose);
        if (!ok) return false;
    }
    return true;
}

template<>
PyObject *pyopencv_from(const ppf_match_3d::Pose3D &pose3D)
{
    return Py_BuildValue("{s:i,s:i,s:d,s:O}",
        "modelIndex", pose3D.modelIndex,
        "numVotes", pose3D.numVotes,
        "residual", pose3D.residual,
        "pose", pyopencv_from(pose3D.pose));
}

#endif
