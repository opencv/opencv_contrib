#ifdef HAVE_OPENCV_STEREO
typedef std::vector<stereo::MatchQuasiDense> vector_MatchQuasiDense;

template<> struct pyopencvVecConverter<stereo::MatchQuasiDense>
{
    static bool to(PyObject* obj, std::vector<stereo::MatchQuasiDense>& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<stereo::MatchQuasiDense>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

#endif
