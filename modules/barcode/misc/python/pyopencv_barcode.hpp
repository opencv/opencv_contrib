#ifdef HAVE_OPENCV_BARCODE
typedef std::vector<barcode::BarcodeType> vector_BarcodeType;

template<> struct pyopencvVecConverter<barcode::BarcodeType>
{
    static bool to(PyObject* obj, std::vector<barcode::BarcodeType>& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }
    static PyObject* from(const std::vector<barcode::BarcodeType>& value)
    {

        return pyopencv_from_generic_vec(value);
    }
};

#endif  // HAVE_OPENCV_BARCODE
